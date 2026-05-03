"""Tests for configuration system.

Covers models, persistence, and manager functionality.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from meetandread.config import (
    AppSettings,
    ConfigManager,
    HardwareSettings,
    ModelSettings,
    SettingsPersistence,
    TranscriptionSettings,
    UISettings,
    get_config,
    get_config_manager,
    save_config,
    set_config,
)


@pytest.fixture
def temp_config_dir():
    """Provide a temporary directory for config files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup
    try:
        for f in Path(temp_dir).glob("*"):
            f.unlink()
        Path(temp_dir).rmdir()
    except OSError:
        pass


@pytest.fixture
def persistence(temp_config_dir):
    """Provide a SettingsPersistence with temp directory."""
    return SettingsPersistence(config_dir=temp_config_dir)


@pytest.fixture
def manager(temp_config_dir):
    """Provide a ConfigManager with temp directory."""
    # Reset singleton for clean test
    ConfigManager._instance = None
    ConfigManager._initialized = False
    
    persistence = SettingsPersistence(config_dir=temp_config_dir)
    cm = ConfigManager(persistence=persistence)
    return cm


# ============================================================================
# Model Tests
# ============================================================================

class TestModelSettings:
    """Tests for ModelSettings dataclass."""
    
    def test_default_values(self):
        """Test default values are correct."""
        settings = ModelSettings()
        assert settings.realtime_model_size == "auto"
    
    def test_to_dict(self):
        """Test serialization to dict."""
        settings = ModelSettings(realtime_model_size="small")
        d = settings.to_dict()
        assert d["realtime_model_size"] == "small"
    
    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {"realtime_model_size": "base"}
        settings = ModelSettings.from_dict(d)
        assert settings.realtime_model_size == "base"
    
    def test_from_dict_missing_fields(self):
        """Test from_dict uses defaults for missing fields."""
        d = {"realtime_model_size": "tiny"}
        settings = ModelSettings.from_dict(d)
        assert settings.realtime_model_size == "tiny"
    
    def test_from_dict_empty_dict(self):
        """Test from_dict with empty dict uses all defaults."""
        settings = ModelSettings.from_dict({})
        assert settings.realtime_model_size == "auto"


class TestTranscriptionSettings:
    """Tests for TranscriptionSettings dataclass."""
    
    def test_default_values(self):
        """Test default values are correct."""
        settings = TranscriptionSettings()
        assert settings.enabled is True
        assert settings.confidence_threshold == 0.7
        assert settings.min_chunk_size_sec == 0.5
        assert settings.agreement_threshold == 1
    
    def test_denoising_default_values(self):
        """Test denoising settings have correct defaults."""
        settings = TranscriptionSettings()
        assert settings.microphone_denoising_enabled is False
        assert settings.microphone_denoising_provider == "spectral_gate"
        assert settings.microphone_denoising_latency_budget_ms == 200
    
    def test_denoising_roundtrip(self):
        """Test denoising settings survive serialization roundtrip."""
        original = TranscriptionSettings(
            microphone_denoising_enabled=False,
            microphone_denoising_provider="some_other_provider",
            microphone_denoising_latency_budget_ms=100,
        )
        d = original.to_dict()
        assert d["microphone_denoising_enabled"] is False
        assert d["microphone_denoising_provider"] == "some_other_provider"
        assert d["microphone_denoising_latency_budget_ms"] == 100
        
        restored = TranscriptionSettings.from_dict(d)
        assert restored.microphone_denoising_enabled is False
        assert restored.microphone_denoising_provider == "some_other_provider"
        assert restored.microphone_denoising_latency_budget_ms == 100
    
    def test_denoising_defaults_from_empty_dict(self):
        """Test denoising settings get defaults from empty dict."""
        settings = TranscriptionSettings.from_dict({})
        assert settings.microphone_denoising_enabled is False
        assert settings.microphone_denoising_provider == "spectral_gate"
        assert settings.microphone_denoising_latency_budget_ms == 200
    
    def test_to_dict_from_dict_roundtrip(self):
        """Test serialization roundtrip."""
        original = TranscriptionSettings(
            enabled=False,
            confidence_threshold=0.8,
            min_chunk_size_sec=2.0,
            agreement_threshold=3
        )
        d = original.to_dict()
        restored = TranscriptionSettings.from_dict(d)
        assert restored.enabled is False
        assert restored.confidence_threshold == 0.8
        assert restored.min_chunk_size_sec == 2.0
        assert restored.agreement_threshold == 3


class TestHardwareSettings:
    """Tests for HardwareSettings dataclass."""
    
    def test_default_values(self):
        """Test default values are correct."""
        settings = HardwareSettings()
        assert settings.auto_detect_on_startup is True
        assert settings.last_detected_ram_gb is None
        assert settings.last_detected_cpu_count is None
        assert settings.recommended_model is None
        assert settings.user_override_model is None


class TestUISettings:
    """Tests for UISettings dataclass."""
    
    def test_default_values(self):
        """Test default values are correct."""
        settings = UISettings()
        assert settings.show_confidence_legend is True
        assert settings.transcript_auto_scroll is True
        assert settings.widget_position is None
        assert settings.widget_dock_edge is None
    
    def test_widget_position_tuple_serialization(self):
        """Test that tuple position is properly serialized."""
        settings = UISettings(widget_position=(100, 200))
        d = settings.to_dict()
        assert d["widget_position"] == [100, 200]  # Serialized as list
        
        restored = UISettings.from_dict(d)
        assert restored.widget_position == (100, 200)  # Restored as tuple
    
    def test_from_dict_with_list_position(self):
        """Test from_dict handles list position."""
        d = {"widget_position": [50, 75]}
        settings = UISettings.from_dict(d)
        assert settings.widget_position == (50, 75)
    
    def test_audio_sources_defaults_to_none(self):
        """Test audio_sources defaults to None."""
        settings = UISettings()
        assert settings.audio_sources is None
    
    def test_audio_sources_roundtrip_single(self):
        """Test audio_sources round-trip serialization with single source."""
        settings = UISettings(audio_sources=['mic'])
        d = settings.to_dict()
        assert d["audio_sources"] == ['mic']
        
        restored = UISettings.from_dict(d)
        assert restored.audio_sources == ['mic']
    
    def test_audio_sources_roundtrip_both(self):
        """Test audio_sources round-trip with both sources."""
        settings = UISettings(audio_sources=['mic', 'system'])
        d = settings.to_dict()
        assert d["audio_sources"] == ['mic', 'system']
        
        restored = UISettings.from_dict(d)
        assert restored.audio_sources == ['mic', 'system']
    
    def test_audio_sources_from_dict_missing_key(self):
        """Test from_dict returns None when audio_sources key is absent."""
        settings = UISettings.from_dict({})
        assert settings.audio_sources is None
    
    def test_audio_sources_from_dict_explicit_none(self):
        """Test from_dict handles explicit None value."""
        settings = UISettings.from_dict({"audio_sources": None})
        assert settings.audio_sources is None


class TestAppSettings:
    """Tests for AppSettings root container."""
    
    def test_default_values(self):
        """Test default values and nested settings."""
        settings = AppSettings()
        assert settings.config_version == 5
        assert isinstance(settings.model, ModelSettings)
        assert isinstance(settings.transcription, TranscriptionSettings)
        assert isinstance(settings.hardware, HardwareSettings)
        assert isinstance(settings.ui, UISettings)
    
    def test_to_dict(self):
        """Test full serialization."""
        settings = AppSettings()
        d = settings.to_dict()
        assert d["config_version"] == 5
        assert "model" in d
        assert "transcription" in d
        assert "hardware" in d
        assert "ui" in d
    
    def test_from_dict(self):
        """Test full deserialization."""
        d = {
            "config_version": 1,
            "model": {"realtime_model_size": "small"},
            "transcription": {"enabled": False},
            "hardware": {"last_detected_ram_gb": 16.0},
            "ui": {"show_confidence_legend": False}
        }
        settings = AppSettings.from_dict(d)
        assert settings.config_version == 1
        assert settings.model.realtime_model_size == "small"
        assert settings.transcription.enabled is False
        assert settings.hardware.last_detected_ram_gb == 16.0
        assert settings.ui.show_confidence_legend is False
    
    def test_from_dict_missing_sections(self):
        """Test from_dict handles missing sections with defaults."""
        d = {"config_version": 1}  # Missing all sections
        settings = AppSettings.from_dict(d)
        assert settings.model.realtime_model_size == "auto"  # default
        assert settings.transcription.enabled is True  # default
    
    def test_from_dict_invalid_sections(self):
        """Test from_dict handles invalid section types."""
        d = {
            "config_version": 1,
            "model": "invalid"  # Should be dict
        }
        settings = AppSettings.from_dict(d)
        # Should use defaults for invalid sections
        assert settings.model.realtime_model_size == "auto"
    
    def test_from_dict_unknown_fields(self):
        """Test from_dict ignores unknown fields gracefully."""
        d = {
            "config_version": 1,
            "unknown_field": "value",
            "model": {"realtime_model_size": "tiny", "unknown_model_field": 123}
        }
        settings = AppSettings.from_dict(d)
        assert settings.config_version == 1
        assert settings.model.realtime_model_size == "tiny"


# ============================================================================
# Persistence Tests
# ============================================================================

class TestSettingsPersistence:
    """Tests for SettingsPersistence."""
    
    def test_get_config_path_creates_directory(self, persistence, temp_config_dir):
        """Test that get_config_path creates the directory."""
        config_path = persistence.get_config_path()
        assert config_path.parent.exists()
        assert config_path.name == "config.json"
    
    def test_save_and_load(self, persistence):
        """Test saving and loading settings."""
        settings = AppSettings()
        settings.model.realtime_model_size = "small"
        
        result = persistence.save_settings(settings)
        assert result is True
        
        loaded = persistence.load_settings()
        assert loaded.model.realtime_model_size == "small"
    
    def test_load_missing_file_returns_defaults(self, persistence):
        """Test loading when file doesn't exist returns defaults."""
        loaded = persistence.load_settings()
        assert loaded.model.realtime_model_size == "auto"
        assert loaded.config_version == 5
    
    def test_load_corrupted_json_returns_defaults(self, persistence):
        """Test loading corrupted file returns defaults."""
        # Create corrupted file
        config_path = persistence.get_config_path()
        config_path.write_text("{invalid json")
        
        loaded = persistence.load_settings()
        assert loaded.model.realtime_model_size == "auto"  # defaults
    
    def test_load_empty_file_returns_defaults(self, persistence):
        """Test loading empty file returns defaults."""
        config_path = persistence.get_config_path()
        config_path.write_text("")
        
        loaded = persistence.load_settings()
        assert loaded.model.realtime_model_size == "auto"
    
    def test_atomic_write(self, persistence, temp_config_dir):
        """Test atomic write doesn't leave temp files."""
        settings = AppSettings()
        persistence.save_settings(settings)
        
        # Should not have any temp files
        temp_files = list(temp_config_dir.glob("config.json.tmp.*"))
        assert len(temp_files) == 0
    
    def test_get_default_settings(self, persistence):
        """Test getting default settings."""
        defaults = persistence.get_default_settings()
        assert defaults.model.realtime_model_size == "auto"
    
    def test_delete_config(self, persistence):
        """Test deleting config file."""
        settings = AppSettings()
        persistence.save_settings(settings)
        
        assert persistence.get_config_path().exists()
        
        result = persistence.delete_config()
        assert result is True
        assert not persistence.get_config_path().exists()
    
    def test_get_config_info(self, persistence):
        """Test getting config info."""
        info = persistence.get_config_info()
        assert "path" in info
        assert "exists" in info
        assert info["exists"] is False  # No file yet
        assert info["current_version"] == 5
    
    def test_config_info_after_save(self, persistence):
        """Test config info after saving."""
        settings = AppSettings()
        persistence.save_settings(settings)
        
        info = persistence.get_config_info()
        assert info["exists"] is True
        assert info["version"] == 5
        assert info["needs_migration"] is False


class TestConfigMigration:
    """Tests for config migration."""
    
    def test_migration_from_version_0(self, persistence):
        """Test migration from pre-versioned config."""
        old_config = {
            "model": {"realtime_model_size": "base"}
            # Missing config_version and other sections
        }
        
        migrated = persistence.migrate_config(old_config, 0)
        assert migrated["config_version"] == 5
        assert "transcription" in migrated
        assert "hardware" in migrated
        assert "ui" in migrated
        # Original values preserved
        assert migrated["model"]["realtime_model_size"] == "base"
    
    def test_migration_from_version_2_adds_denoising(self, persistence):
        """Test migration from version 2 adds denoising settings."""
        old_config = {
            "config_version": 2,
            "model": {"realtime_model_size": "small"},
            "transcription": {
                "enabled": True,
                "confidence_threshold": 0.8,
                "benchmark_history": {"base": {"wer": 0.17}},
            },
        }
        
        migrated = persistence.migrate_config(old_config, 2)
        assert migrated["config_version"] == 5
        
        # Denoising defaults added (disabled by default)
        ts = migrated["transcription"]
        assert ts["microphone_denoising_enabled"] is False
        assert ts["microphone_denoising_provider"] == "spectral_gate"
        assert ts["microphone_denoising_latency_budget_ms"] == 200
        
        # Existing values preserved
        assert ts["enabled"] is True
        assert ts["confidence_threshold"] == 0.8
        assert ts["benchmark_history"] == {"base": {"wer": 0.17}}
        assert migrated["model"]["realtime_model_size"] == "small"

        # Speaker min-duration defaults added by v3→v4 migration
        speaker = migrated["speaker"]
        assert speaker["min_duration_on"] == 0.3
        assert speaker["min_duration_off"] == 0.8
    
    def test_migration_preserves_existing_denoising_values(self, persistence):
        """Test migration doesn't clobber existing denoising fields."""
        old_config = {
            "config_version": 2,
            "transcription": {
                "microphone_denoising_enabled": False,
                "microphone_denoising_provider": "custom",
                "microphone_denoising_latency_budget_ms": 50,
            },
        }
        
        migrated = persistence.migrate_config(old_config, 2)
        ts = migrated["transcription"]
        assert ts["microphone_denoising_enabled"] is False
        assert ts["microphone_denoising_provider"] == "custom"
        assert ts["microphone_denoising_latency_budget_ms"] == 50
    
    def test_migration_from_version_3_adds_speaker_min_duration(self, persistence):
        """Test v3→v4→v5 migration adds min_duration_on/off and disables denoising."""
        old_config = {
            "config_version": 3,
            "speaker": {
                "enabled": True,
                "confidence_threshold": 0.7,
                "clustering_threshold": 0.6,
            },
        }

        migrated = persistence.migrate_config(old_config, 3)
        assert migrated["config_version"] == 5

        speaker = migrated["speaker"]
        assert speaker["min_duration_on"] == 0.3
        assert speaker["min_duration_off"] == 0.8
        # Existing values preserved
        assert speaker["enabled"] is True
        assert speaker["confidence_threshold"] == 0.7
        assert speaker["clustering_threshold"] == 0.6
        # v4→v5 disables denoising
        assert migrated["transcription"]["microphone_denoising_enabled"] is False

    def test_migration_from_version_3_preserves_existing_min_duration(self, persistence):
        """Test v3→v4 migration doesn't clobber existing min_duration fields."""
        old_config = {
            "config_version": 3,
            "speaker": {
                "enabled": False,
                "min_duration_on": 0.5,
                "min_duration_off": 1.2,
            },
        }

        migrated = persistence.migrate_config(old_config, 3)
        speaker = migrated["speaker"]
        assert speaker["min_duration_on"] == 0.5
        assert speaker["min_duration_off"] == 1.2
        assert speaker["enabled"] is False

    def test_migration_from_version_3_missing_speaker_section(self, persistence):
        """Test v3→v4 migration handles missing speaker section."""
        old_config = {
            "config_version": 3,
        }

        migrated = persistence.migrate_config(old_config, 3)
        speaker = migrated["speaker"]
        assert speaker["min_duration_on"] == 0.3
        assert speaker["min_duration_off"] == 0.8
    
    def test_no_migration_needed(self, persistence):
        """Test config at current version doesn't change."""
        current_config = {
            "config_version": 5,
            "model": {"realtime_model_size": "small"}
        }
        
        migrated = persistence.migrate_config(current_config, 5)
        assert migrated["config_version"] == 5


# ============================================================================
# Manager Tests
# ============================================================================

class TestConfigManager:
    """Tests for ConfigManager."""
    
    def test_get_entire_settings(self, manager):
        """Test get() with no key_path returns entire settings."""
        settings = manager.get()
        assert isinstance(settings, AppSettings)
    
    def test_get_nested_value(self, manager):
        """Test get() with dot-path."""
        value = manager.get("model.realtime_model_size")
        assert value == "auto"
    
    def test_get_deeply_nested(self, manager):
        """Test get() with deeply nested path."""
        value = manager.get("transcription.confidence_threshold")
        assert value == 0.7
    
    def test_get_invalid_path(self, manager):
        """Test get() with invalid path raises ValueError."""
        with pytest.raises(ValueError):
            manager.get("invalid.path")
        
        with pytest.raises(ValueError):
            manager.get("model.nonexistent")
    
    def test_set_value(self, manager):
        """Test set() changes value."""
        manager.set("model.realtime_model_size", "small")
        assert manager.get("model.realtime_model_size") == "small"
    
    def test_set_marks_dirty(self, manager):
        """Test set() marks as dirty."""
        assert not manager.is_dirty()
        manager.set("model.realtime_model_size", "small")
        assert manager.is_dirty()
    
    def test_set_type_validation(self, manager):
        """Test set() validates types."""
        # Valid type
        manager.set("transcription.enabled", False)
        
        # Invalid type - should raise ValueError
        with pytest.raises(ValueError):
            manager.set("transcription.enabled", "not_a_bool")
        
        with pytest.raises(ValueError):
            manager.set("transcription.confidence_threshold", "not_a_number")
    
    def test_set_invalid_path(self, manager):
        """Test set() with invalid path raises ValueError."""
        with pytest.raises(ValueError):
            manager.set("invalid", "value")
        
        with pytest.raises(ValueError):
            manager.set("model.nonexistent", "value")
    
    def test_get_dirty_paths(self, manager):
        """Test getting list of dirty paths."""
        manager.set("model.realtime_model_size", "small")
        manager.set("transcription.enabled", False)
        
        dirty = manager.get_dirty_paths()
        assert "model.realtime_model_size" in dirty
        assert "transcription.enabled" in dirty
    
    def test_save_clears_dirty(self, manager):
        """Test save() clears dirty flag."""
        manager.set("model.realtime_model_size", "small")
        assert manager.is_dirty()
        
        manager.save()
        assert not manager.is_dirty()
    
    def test_save_not_dirty(self, manager):
        """Test save() when not dirty returns True."""
        result = manager.save()
        assert result is True
    
    def test_reload_clears_dirty(self, manager):
        """Test reload() clears dirty flag."""
        manager.set("model.realtime_model_size", "small")
        manager.reload()
        assert not manager.is_dirty()
        # Value should be back to default
        assert manager.get("model.realtime_model_size") == "auto"
    
    def test_reset_to_defaults(self, manager):
        """Test reset_to_defaults()."""
        manager.set("model.realtime_model_size", "small")
        manager.set("transcription.enabled", False)
        manager.save()
        
        manager.reset_to_defaults()
        assert manager.get("model.realtime_model_size") == "auto"
        assert manager.get("transcription.enabled") is True
        assert manager.is_dirty()  # All paths marked dirty
    
    def test_get_config_path(self, manager, temp_config_dir):
        """Test get_config_path()."""
        path = manager.get_config_path()
        assert str(temp_config_dir) in path
        assert path.endswith("config.json")
    
    def test_get_config_info(self, manager):
        """Test get_config_info()."""
        info = manager.get_config_info()
        assert "path" in info
        assert "is_dirty" in info
        assert "dirty_paths" in info
        assert not info["is_dirty"]


class TestConfigManagerSingleton:
    """Tests for ConfigManager singleton behavior."""
    
    def test_singleton_instance(self, temp_config_dir):
        """Test that multiple creations return same instance."""
        # Reset singleton
        ConfigManager._instance = None
        ConfigManager._initialized = False
        
        persistence = SettingsPersistence(config_dir=temp_config_dir)
        cm1 = ConfigManager(persistence=persistence)
        cm2 = ConfigManager(persistence=persistence)
        
        assert cm1 is cm2
    
    def test_module_level_get_config_manager(self, temp_config_dir):
        """Test module-level get_config_manager()."""
        # Reset singleton
        ConfigManager._instance = None
        ConfigManager._initialized = False
        
        persistence = SettingsPersistence(config_dir=temp_config_dir)
        cm = ConfigManager(persistence=persistence)
        
        # Should return same instance
        cm2 = get_config_manager()
        assert cm is cm2


# ============================================================================
# Integration Tests
# ============================================================================

class TestConfigIntegration:
    """Integration tests for full config system."""
    
    def test_full_workflow(self, temp_config_dir):
        """Test complete get/set/save workflow."""
        # Reset singleton
        ConfigManager._instance = None
        ConfigManager._initialized = False
        
        persistence = SettingsPersistence(config_dir=temp_config_dir)
        cm = ConfigManager(persistence=persistence)
        
        # Initial state
        assert cm.get("model.realtime_model_size") == "auto"
        assert not cm.is_dirty()
        
        # Modify
        cm.set("model.realtime_model_size", "small")
        assert cm.is_dirty()
        
        # Save
        cm.save()
        assert not cm.is_dirty()
        
        # Verify file exists
        assert persistence.get_config_path().exists()
        
        # Create new manager (simulating app restart)
        ConfigManager._instance = None
        ConfigManager._initialized = False
        
        persistence2 = SettingsPersistence(config_dir=temp_config_dir)
        cm2 = ConfigManager(persistence=persistence2)
        
        # Should load saved value
        assert cm2.get("model.realtime_model_size") == "small"
    
    def test_multiple_saves_no_corruption(self, temp_config_dir):
        """Test multiple saves don't corrupt file."""
        persistence = SettingsPersistence(config_dir=temp_config_dir)
        
        for i in range(10):
            settings = AppSettings()
            settings.model.realtime_model_size = f"small_{i}"
            persistence.save_settings(settings)
        
        loaded = persistence.load_settings()
        assert "small_9" in loaded.model.realtime_model_size
    
    def test_convenience_functions(self, temp_config_dir):
        """Test module-level convenience functions."""
        import meetandread.config.manager as manager_module
        
        # Reset singleton and module-level instance
        ConfigManager._instance = None
        ConfigManager._initialized = False
        manager_module._config_manager = None
        
        # Create persistence first
        persistence = SettingsPersistence(config_dir=temp_config_dir)
        
        # Create ConfigManager with explicit persistence
        cm = ConfigManager(persistence=persistence)
        
        # Also set the module-level reference
        manager_module._config_manager = cm
        
        # Use convenience functions
        assert get_config("model.realtime_model_size") == "auto"
        
        set_config("model.realtime_model_size", "base")
        assert get_config("model.realtime_model_size") == "base"
        
        result = save_config()
        assert result is True
        
        # Verify persistence
        assert persistence.get_config_path().exists()


class TestConfigConcurrency:
    """Tests for concurrent access patterns."""
    
    def test_save_while_dirty(self, manager):
        """Test saving multiple times while dirty."""
        manager.set("model.realtime_model_size", "small")
        assert manager.save() is True
        
        # Change again
        manager.set("model.realtime_model_size", "base")
        assert manager.save() is True
        
        # Reload and verify
        manager.reload()
        assert manager.get("model.realtime_model_size") == "base"


# ============================================================================
# Edge Cases
# ============================================================================

class TestConfigEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_none_values(self, manager):
        """Test handling of None values."""
        # None should be valid for Optional fields
        manager.set("hardware.user_override_model", None)
        assert manager.get("hardware.user_override_model") is None
    
    def test_widget_position_none(self, manager):
        """Test widget position can be None."""
        manager.set("ui.widget_position", None)
        assert manager.get("ui.widget_position") is None
    
    def test_numeric_type_flexibility(self, manager):
        """Test int/float interchangeability."""
        # Setting int on float field should work
        manager.set("transcription.confidence_threshold", 1)
        assert manager.get("transcription.confidence_threshold") == 1
        
        # Setting float on int field should work
        manager.set("transcription.agreement_threshold", 2.0)
        assert manager.get("transcription.agreement_threshold") == 2.0
    
    def test_denoising_enabled_get_set(self, manager):
        """Test get/set for microphone_denoising_enabled."""
        assert manager.get("transcription.microphone_denoising_enabled") is False
        manager.set("transcription.microphone_denoising_enabled", True)
        assert manager.get("transcription.microphone_denoising_enabled") is True
    
    def test_denoising_provider_get_set(self, manager):
        """Test get/set for microphone_denoising_provider."""
        assert manager.get("transcription.microphone_denoising_provider") == "spectral_gate"
        manager.set("transcription.microphone_denoising_provider", "other")
        assert manager.get("transcription.microphone_denoising_provider") == "other"
    
    def test_denoising_budget_get_set(self, manager):
        """Test get/set for microphone_denoising_latency_budget_ms."""
        assert manager.get("transcription.microphone_denoising_latency_budget_ms") == 200
        manager.set("transcription.microphone_denoising_latency_budget_ms", 100)
        assert manager.get("transcription.microphone_denoising_latency_budget_ms") == 100
    
    def test_denoising_budget_type_validation(self, manager):
        """Test type validation for denoising budget."""
        with pytest.raises(ValueError):
            manager.set("transcription.microphone_denoising_latency_budget_ms", "not_a_number")
    
    def test_empty_string_path(self, manager):
        """Test empty string path is invalid."""
        with pytest.raises(ValueError):
            manager.set("", "value")
    
    def test_single_part_path(self, manager):
        """Test single part path is invalid."""
        with pytest.raises(ValueError):
            manager.set("model", "value")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
