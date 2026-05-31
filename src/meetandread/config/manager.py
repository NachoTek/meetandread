"""ConfigManager - Main API for settings management.

Provides ConfigManager class as the primary interface for getting,
setting, and saving application settings with smart defaults tracking.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from meetandread.config.models import (
    AppSettings,
    StoragePaths,
)
from meetandread.config.persistence import SettingsPersistence


logger = logging.getLogger(__name__)


def validate_storage_paths(paths: StoragePaths) -> Dict[str, str]:
    """Validate custom storage paths by expanding, creating, and write-testing.

    For each non-None path in *paths*, the function:
    1. Expands ``~`` and environment variables via ``Path.expanduser()`` /
       ``Path.resolve()``.
    2. Creates the directory if it doesn't exist.
    3. Writes and removes a temporary sentinel file to confirm write access.
    4. Returns field-specific error messages on failure.

    Returns a dict mapping field names to error strings — empty means all OK.
    Only field names and generic error types are logged (no raw paths or content).
    """
    errors: Dict[str, str] = {}

    field_map: List[Tuple[str, Optional[str]]] = [
        ("transcripts_path", paths.transcripts_path),
        ("recordings_path", paths.recordings_path),
        ("logs_path", paths.logs_path),
    ]

    for field_name, raw_value in field_map:
        if raw_value is None:
            continue
        try:
            resolved = Path(raw_value).expanduser().resolve()
        except Exception:
            errors[field_name] = "Path could not be resolved"
            logger.warning("storage_path_validation_failed field=%s reason=resolve_error", field_name)
            continue

        try:
            resolved.mkdir(parents=True, exist_ok=True)
        except OSError:
            errors[field_name] = "Directory could not be created"
            logger.warning("storage_path_validation_failed field=%s reason=mkdir_error", field_name)
            continue

        # Write-test with a sentinel file
        try:
            sentinel = resolved / f".meetandread_write_test_{id(paths)}"
            sentinel.write_text("ok", encoding="utf-8")
            sentinel.unlink(missing_ok=True)
        except OSError:
            errors[field_name] = "Directory is not writable"
            logger.warning("storage_path_validation_failed field=%s reason=write_test_error", field_name)
            continue

    return errors


class ConfigManager:
    """Main API for managing application configuration.
    
    Provides a clean interface for:
    - Getting/setting specific configuration values via dot-path notation
    - Tracking which settings have been modified (dirty tracking)
    - Persisting only changed settings (smart defaults)
    - Resetting to defaults
    
    The ConfigManager uses a singleton pattern - one instance per application.
    Settings auto-load on first access and can be persisted with save().
    
    Example:
        >>> from meetandread.config.manager import ConfigManager
        >>> cm = ConfigManager()
        >>> cm.get('transcription.realtime_model_size')
        'auto'
        >>> cm.set('transcription.realtime_model_size', 'small')
        >>> cm.save()
    """
    
    _instance: Optional["ConfigManager"] = None
    _initialized: bool = False
    
    def __new__(cls, *args, **kwargs) -> "ConfigManager":
        """Singleton pattern - ensure only one ConfigManager exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, persistence: Optional[SettingsPersistence] = None):
        """Initialize the ConfigManager.
        
        Only initializes on first creation (singleton pattern).
        
        Args:
            persistence: Optional custom persistence instance. If None,
                creates default SettingsPersistence.
        """
        if ConfigManager._initialized:
            return
        
        self._persistence = persistence or SettingsPersistence()
        
        # Load settings or use defaults
        try:
            self._settings = self._persistence.load_settings()
            logger.info(f"Config loaded from {self._persistence.get_config_path()}")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
            self._settings = AppSettings.get_defaults()
        
        # Store reference to original defaults for smart defaults tracking
        self._defaults = AppSettings.get_defaults()
        
        # Track which paths have been modified
        self._dirty_paths: set = set()
        
        ConfigManager._initialized = True
    
    def get_settings(self) -> AppSettings:
        """Get the current settings object.
        
        Returns:
            Current AppSettings instance.
        """
        return self._settings
    
    def get(self, key_path: Optional[str] = None) -> Any:
        """Get a specific setting by dot-path notation.
        
        Args:
            key_path: Dot-separated path to setting (e.g., "transcription.realtime_model_size").
                If None, returns the entire AppSettings object.
        
        Returns:
            The setting value, or AppSettings if key_path is None.
        
        Raises:
            ValueError: If key_path is invalid or setting doesn't exist.
        
        Example:
            >>> cm.get('transcription.realtime_model_size')
            'auto'
            >>> cm.get('transcription.enabled')
            True
            >>> cm.get()  # Returns entire AppSettings
            AppSettings(...)
        """
        if key_path is None:
            return self._settings
        
        parts = key_path.split('.')
        
        # Navigate to the correct object
        current: Any = self._settings
        
        for part in parts:
            if not hasattr(current, part):
                raise ValueError(f"Invalid key path: '{key_path}' - '{part}' not found")
            current = getattr(current, part)
        
        return current
    
    def set(self, key_path: str, value: Any) -> None:
        """Set a specific setting by dot-path notation.
        
        Marks the setting as dirty (modified from defaults).
        Performs basic type validation.
        
        Args:
            key_path: Dot-separated path to setting (e.g., "transcription.realtime_model_size").
            value: Value to set.
        
        Raises:
            ValueError: If key_path is invalid, setting doesn't exist,
                or value has wrong type.
        
        Example:
            >>> cm.set('transcription.realtime_model_size', 'small')
            >>> cm.set('transcription.enabled', False)
        """
        parts = key_path.split('.')
        
        if len(parts) < 2:
            raise ValueError(f"Invalid key path: '{key_path}' - must have at least 2 parts (e.g., 'transcription.realtime_model_size')")
        
        # Navigate to the parent object
        parent: Any = self._settings
        for part in parts[:-1]:
            if not hasattr(parent, part):
                raise ValueError(f"Invalid key path: '{key_path}' - '{part}' not found")
            parent = getattr(parent, part)
        
        target_attr = parts[-1]
        
        if not hasattr(parent, target_attr):
            raise ValueError(f"Invalid key path: '{key_path}' - '{target_attr}' not found")
        
        # Get current value for type checking
        current_value = getattr(parent, target_attr)
        
        # Type validation
        if current_value is not None and value is not None:
            expected_type = type(current_value)
            
            # Special handling for Optional types
            if expected_type in (int, float):
                # Allow int/float interchangeability for numeric fields
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Invalid type for '{key_path}': expected numeric, got {type(value).__name__}")
            elif expected_type == bool:
                # Strict bool checking (don't accept truthy/falsy values)
                if not isinstance(value, bool):
                    raise ValueError(f"Invalid type for '{key_path}': expected bool, got {type(value).__name__}")
            elif not isinstance(value, expected_type):
                # Allow int for float fields, but otherwise strict
                raise ValueError(f"Invalid type for '{key_path}': expected {expected_type.__name__}, got {type(value).__name__}")
        
        # Set the value
        setattr(parent, target_attr, value)
        
        # Mark as dirty
        self._dirty_paths.add(key_path)
        
        logger.debug(f"Setting '{key_path}' = {value}")
    
    def is_dirty(self) -> bool:
        """Check if any settings have been modified from defaults.
        
        Returns:
            True if settings need saving, False otherwise.
        """
        return len(self._dirty_paths) > 0
    
    def get_dirty_paths(self) -> List[str]:
        """Get list of setting paths that have been modified.
        
        Returns:
            List of dot-paths that have been changed.
        """
        return sorted(list(self._dirty_paths))
    
    def save(self) -> bool:
        """Save settings if dirty.
        
        Only persists settings if they have been modified from defaults.
        After successful save, clears dirty flag.
        
        Returns:
            True if saved successfully or not dirty, False on error.
        """
        if not self.is_dirty():
            logger.debug("Settings not dirty, skipping save")
            return True
        
        result = self._persistence.save_settings(self._settings)
        
        if result:
            self._dirty_paths.clear()
            logger.info("Settings saved successfully")
        else:
            logger.error("Failed to save settings")
        
        return result
    
    def reset_to_defaults(self) -> None:
        """Reset all settings to their default values.
        
        Marks all settings as dirty so they will be persisted.
        """
        self._settings = AppSettings.get_defaults()
        
        # Mark all paths as dirty so they get saved
        self._dirty_paths = self._get_all_paths()
        
        logger.info("Settings reset to defaults")
    
    def _get_all_paths(self) -> set:
        """Get all setting paths for tracking purposes.
        
        Returns:
            Set of all possible dot-paths.
        """
        return {
            # Transcription settings (realtime_model_size moved here from ModelSettings)
            "transcription.realtime_model_size",
            # Transcription settings
            "transcription.enabled",
            "transcription.confidence_threshold",
            "transcription.min_chunk_size_sec",
            "transcription.agreement_threshold",
            # Hardware settings
            "hardware.auto_detect_on_startup",
            "hardware.last_detected_ram_gb",
            "hardware.last_detected_cpu_count",
            "hardware.recommended_model",
            "hardware.user_override_model",
            # UI settings
            "ui.show_confidence_legend",
            "ui.transcript_auto_scroll",
            "ui.widget_position",
            "ui.widget_dock_edge",
            "ui.audio_sources",
            "ui.waveform_enabled",
            # Transcription denoising settings
            "transcription.microphone_denoising_enabled",
            "transcription.microphone_denoising_provider",
            "transcription.microphone_denoising_latency_budget_ms",
            # Storage path settings
            "storage_paths.transcripts_path",
            "storage_paths.recordings_path",
            "storage_paths.logs_path",
        }
    
    def get_config_path(self) -> str:
        """Get the config file path.
        
        Returns:
            Path to the config file as string.
        """
        return str(self._persistence.get_config_path())
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get information about the current config state.
        
        Returns:
            Dictionary with config metadata.
        """
        info = self._persistence.get_config_info()
        info["is_dirty"] = self.is_dirty()
        info["dirty_paths"] = self.get_dirty_paths()
        return info
    
    def reload(self) -> None:
        """Reload settings from disk.
        
        Discards any unsaved changes and reloads from file.
        """
        self._settings = self._persistence.load_settings()
        self._dirty_paths.clear()
        logger.info("Settings reloaded from disk")


def get_config_manager() -> ConfigManager:
    """Get the singleton ConfigManager instance.
    
    Creates the instance on first call.
    
    Returns:
        ConfigManager singleton instance.
    """
    if ConfigManager._instance is None:
        ConfigManager()
    return ConfigManager._instance


def get_config(key_path: Optional[str] = None) -> Any:
    """Convenience function to get a config value.
    
    Args:
        key_path: Dot-separated path to setting. If None, returns entire settings.
    
    Returns:
        Config value or AppSettings.
    
    Example:
        >>> from meetandread.config import get_config
        >>> model_size = get_config('transcription.realtime_model_size')
    """
    return get_config_manager().get(key_path)


def set_config(key_path: str, value: Any) -> None:
    """Convenience function to set a config value.
    
    Args:
        key_path: Dot-separated path to setting.
        value: Value to set.
    
    Example:
        >>> from meetandread.config import set_config
        >>> set_config('transcription.realtime_model_size', 'small')
    """
    get_config_manager().set(key_path, value)


def save_config() -> bool:
    """Convenience function to save config if dirty.
    
    Returns:
        True if saved or not dirty, False on error.
    
    Example:
        >>> from meetandread.config import save_config
        >>> save_config()
    """
    return get_config_manager().save()
