"""Configuration module for meetandread.

Provides type-safe settings management with JSON persistence, versioning,
and smart defaults. Only user-modified settings are persisted.

Quick Start:
    >>> from meetandread.config import get_config, set_config, save_config
    >>>
    >>> # Get model size
    >>> model_size = get_config('model.realtime_model_size')
    >>>
    >>> # Change setting
    >>> set_config('model.realtime_model_size', 'base')
    >>> save_config()

Settings Categories:
    - model: Model selection (realtime_model_size)
    - transcription: Transcription behavior (enabled, confidence_threshold, etc.)
    - hardware: Hardware detection and recommendations
    - ui: UI behavior and appearance

Classes:
    AppSettings: Root container for all settings.
    ModelSettings: Model size configuration.
    TranscriptionSettings: Transcription behavior.
    HardwareSettings: Hardware detection settings.
    UISettings: UI appearance and behavior.
    ConfigManager: Main API for getting/setting/saving.
    SettingsPersistence: Low-level JSON persistence.

Functions:
    get_config: Get a config value or entire settings.
    set_config: Set a config value.
    save_config: Save config if modified.
"""

from meetandread.config.manager import (
    ConfigManager,
    get_config,
    get_config_manager,
    save_config,
    set_config,
    validate_storage_paths,
)
from meetandread.config.models import (
    AppSettings,
    HardwareSettings,
    ModelSettings,
    StoragePaths,
    TranscriptionSettings,
    UISettings,
)
from meetandread.config.persistence import (
    CURRENT_CONFIG_VERSION,
    ConfigVersion,
    SettingsPersistence,
)

__all__ = [
    # Models
    "AppSettings",
    "ModelSettings",
    "TranscriptionSettings",
    "HardwareSettings",
    "UISettings",
    "StoragePaths",
    # Manager
    "ConfigManager",
    "get_config_manager",
    # Validation
    "validate_storage_paths",
    # Convenience functions
    "get_config",
    "set_config",
    "save_config",
    # Persistence
    "SettingsPersistence",
    "ConfigVersion",
    "CURRENT_CONFIG_VERSION",
]
