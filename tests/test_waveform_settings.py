"""Tests for ui.waveform_enabled config contract.

Covers model field, persistence migration, save/reload round-trip,
type validation, and edge cases per T01 must-haves and negative tests.
"""

import json
import tempfile
from pathlib import Path

import pytest

from unittest.mock import patch

from meetandread.config import (
    AppSettings,
    ConfigManager,
    SettingsPersistence,
    UISettings,
    set_config,
    get_config,
    save_config,
)


@pytest.fixture
def temp_config_dir():
    """Provide a temporary directory for config files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
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
    ConfigManager._instance = None
    ConfigManager._initialized = False
    persistence = SettingsPersistence(config_dir=temp_config_dir)
    cm = ConfigManager(persistence=persistence)
    return cm


# ============================================================================
# Model tests
# ============================================================================

class TestWaveformEnabledModel:
    """Tests for waveform_enabled field on UISettings."""

    def test_default_is_true(self):
        """Newly created UISettings defaults waveform_enabled to True."""
        ui = UISettings()
        assert ui.waveform_enabled is True

    def test_from_dict_missing_key_returns_true(self):
        """Missing waveform_enabled key in dict defaults to True."""
        ui = UISettings.from_dict({})
        assert ui.waveform_enabled is True

    def test_from_dict_explicit_true(self):
        """Explicit True survives from_dict."""
        ui = UISettings.from_dict({"waveform_enabled": True})
        assert ui.waveform_enabled is True

    def test_from_dict_explicit_false(self):
        """Explicit False survives from_dict."""
        ui = UISettings.from_dict({"waveform_enabled": False})
        assert ui.waveform_enabled is False

    def test_roundtrip_true(self):
        """True survives to_dict/from_dict round-trip."""
        ui = UISettings(waveform_enabled=True)
        d = ui.to_dict()
        assert d["waveform_enabled"] is True
        restored = UISettings.from_dict(d)
        assert restored.waveform_enabled is True

    def test_roundtrip_false(self):
        """False survives to_dict/from_dict round-trip."""
        ui = UISettings(waveform_enabled=False)
        d = ui.to_dict()
        assert d["waveform_enabled"] is False
        restored = UISettings.from_dict(d)
        assert restored.waveform_enabled is False

    def test_preserves_other_ui_fields(self):
        """Setting waveform_enabled doesn't affect other UI fields."""
        ui = UISettings(
            audio_sources=["mic", "system"],
            cc_panel_geometry=(10, 20, 400, 300),
            show_confidence_legend=False,
            waveform_enabled=False,
        )
        d = ui.to_dict()
        restored = UISettings.from_dict(d)
        assert restored.audio_sources == ["mic", "system"]
        assert restored.cc_panel_geometry == (10, 20, 400, 300)
        assert restored.show_confidence_legend is False
        assert restored.waveform_enabled is False


# ============================================================================
# AppSettings integration
# ============================================================================

class TestWaveformEnabledAppSettings:
    """Tests for waveform_enabled at the AppSettings level."""

    def test_default_app_settings_has_waveform_enabled(self):
        """Fresh AppSettings includes waveform_enabled=True."""
        settings = AppSettings()
        assert settings.ui.waveform_enabled is True

    def test_from_dict_without_ui_section(self):
        """Missing ui section still yields waveform_enabled=True."""
        settings = AppSettings.from_dict({})
        assert settings.ui.waveform_enabled is True

    def test_from_dict_with_ui_but_no_waveform_key(self):
        """UI section present but without waveform_enabled defaults to True."""
        settings = AppSettings.from_dict({"ui": {"show_confidence_legend": False}})
        assert settings.ui.waveform_enabled is True
        assert settings.ui.show_confidence_legend is False


# ============================================================================
# Migration tests
# ============================================================================

class TestWaveformEnabledMigration:
    """Tests for v6→v7 migration adding waveform_enabled."""

    def test_migration_from_v6_adds_waveform_enabled(self, persistence):
        """Migrating a v6 config adds waveform_enabled=True to ui."""
        old_config = {
            "config_version": 6,
            "model": {"realtime_model_size": "base"},
            "ui": {
                "show_confidence_legend": False,
                "audio_sources": ["mic"],
            },
        }
        migrated = persistence.migrate_config(old_config, 6)
        assert migrated["config_version"] == 7
        assert migrated["ui"]["waveform_enabled"] is True
        # Existing values preserved
        assert migrated["ui"]["show_confidence_legend"] is False
        assert migrated["ui"]["audio_sources"] == ["mic"]

    def test_migration_preserves_existing_waveform_enabled(self, persistence):
        """If waveform_enabled already set, migration doesn't overwrite."""
        old_config = {
            "config_version": 6,
            "ui": {"waveform_enabled": False},
        }
        migrated = persistence.migrate_config(old_config, 6)
        assert migrated["ui"]["waveform_enabled"] is False

    def test_migration_handles_missing_ui_section(self, persistence):
        """Migration works when ui section is entirely missing."""
        old_config = {
            "config_version": 6,
            "model": {"realtime_model_size": "tiny"},
        }
        migrated = persistence.migrate_config(old_config, 6)
        assert migrated["ui"]["waveform_enabled"] is True

    def test_migration_does_not_overwrite_audio_sources(self, persistence):
        """Migration preserves audio_sources untouched."""
        old_config = {
            "config_version": 6,
            "ui": {"audio_sources": ["system"]},
        }
        migrated = persistence.migrate_config(old_config, 6)
        assert migrated["ui"]["audio_sources"] == ["system"]
        assert migrated["ui"]["waveform_enabled"] is True

    def test_migration_does_not_overwrite_cc_panel_geometry(self, persistence):
        """Migration preserves cc_panel_geometry untouched."""
        old_config = {
            "config_version": 6,
            "ui": {"cc_panel_geometry": [100, 200, 640, 480]},
        }
        migrated = persistence.migrate_config(old_config, 6)
        assert migrated["ui"]["cc_panel_geometry"] == [100, 200, 640, 480]
        assert migrated["ui"]["waveform_enabled"] is True


# ============================================================================
# Persistence save/reload tests
# ============================================================================

class TestWaveformEnabledPersistence:
    """Tests for waveform_enabled surviving save/reload."""

    def test_save_reload_true(self, persistence):
        """waveform_enabled=True survives save/reload."""
        settings = AppSettings()
        assert settings.ui.waveform_enabled is True
        persistence.save_settings(settings)
        loaded = persistence.load_settings()
        assert loaded.ui.waveform_enabled is True

    def test_save_reload_false(self, persistence):
        """waveform_enabled=False survives save/reload."""
        settings = AppSettings()
        settings.ui.waveform_enabled = False
        persistence.save_settings(settings)
        loaded = persistence.load_settings()
        assert loaded.ui.waveform_enabled is False

    def test_v6_file_migrated_on_load(self, persistence):
        """Loading a v6 config file migrates it and waveform_enabled defaults True."""
        config_path = persistence.get_config_path()
        v6_config = {
            "config_version": 6,
            "model": {"realtime_model_size": "auto"},
            "ui": {"show_confidence_legend": True},
        }
        config_path.write_text(json.dumps(v6_config, indent=2))
        loaded = persistence.load_settings()
        assert loaded.config_version == 7
        assert loaded.ui.waveform_enabled is True
        assert loaded.ui.show_confidence_legend is True


# ============================================================================
# Manager get/set tests
# ============================================================================

class TestWaveformEnabledManager:
    """Tests for waveform_enabled via ConfigManager."""

    def test_get_default(self, manager):
        """Default value is True via manager."""
        assert manager.get("ui.waveform_enabled") is True

    def test_set_false(self, manager):
        """Can set waveform_enabled to False."""
        manager.set("ui.waveform_enabled", False)
        assert manager.get("ui.waveform_enabled") is False

    def test_set_true(self, manager):
        """Can set waveform_enabled to True."""
        manager.set("ui.waveform_enabled", False)
        manager.set("ui.waveform_enabled", True)
        assert manager.get("ui.waveform_enabled") is True

    def test_type_validation_rejects_non_bool(self, manager):
        """Setting a non-bool value raises ValueError. Note: None bypasses
        validation by design in the existing ConfigManager to support
        Optional fields — this is consistent with other bool fields."""
        with pytest.raises(ValueError, match="expected bool"):
            manager.set("ui.waveform_enabled", "yes")
        with pytest.raises(ValueError, match="expected bool"):
            manager.set("ui.waveform_enabled", 1)

    def test_set_persists_after_save_reload(self, manager, temp_config_dir):
        """waveform_enabled=False persists across save and reload."""
        manager.set("ui.waveform_enabled", False)
        manager.save()

        # Simulate app restart
        ConfigManager._instance = None
        ConfigManager._initialized = False
        p2 = SettingsPersistence(config_dir=temp_config_dir)
        cm2 = ConfigManager(persistence=p2)
        assert cm2.get("ui.waveform_enabled") is False

    def test_in_all_paths(self, manager):
        """ui.waveform_enabled is included in _get_all_paths()."""
        all_paths = manager._get_all_paths()
        assert "ui.waveform_enabled" in all_paths


# ============================================================================
# T02: Settings handler tests
# ============================================================================

class TestWaveformToggleHandler:
    """Tests for _on_waveform_toggled handler in FloatingSettingsPanel."""

    @pytest.fixture
    def qapp_local(self):
        """Provide a QApplication for the test session."""
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app

    @pytest.fixture
    def panel_and_mocks(self, qapp_local, manager):
        """Create a FloatingSettingsPanel with mocked config dependencies."""
        from unittest.mock import MagicMock, patch
        from meetandread.widgets.floating_panels import FloatingSettingsPanel

        fake_screen = MagicMock()
        fake_screen.geometry.return_value = MagicMock(
            x=lambda: 0, y=lambda: 0, width=lambda: 1920, height=lambda: 1080
        )
        with patch("meetandread.config.get_config", return_value=True), \
             patch("meetandread.config.save_config"), \
             patch("meetandread.widgets.floating_panels.QApplication.primaryScreen",
                    return_value=fake_screen), \
             patch("meetandread.widgets.floating_panels.QApplication.screens",
                    return_value=[fake_screen]):
            panel = FloatingSettingsPanel(
                None,
                controller=MagicMock(),
                tray_manager=None,
                main_widget=None,
            )
        return panel

    def test_handler_persists_true(self, panel_and_mocks, manager):
        """Checked state (state=2) persists ui.waveform_enabled=True."""
        from PyQt6.QtCore import Qt
        panel = panel_and_mocks
        with patch("meetandread.config.set_config") as mock_set, \
             patch("meetandread.config.save_config") as mock_save:
            panel._on_waveform_toggled(Qt.CheckState.Checked.value)
            mock_set.assert_called_once_with("ui.waveform_enabled", True)
            mock_save.assert_called_once()

    def test_handler_persists_false(self, panel_and_mocks, manager):
        """Unchecked state (state=0) persists ui.waveform_enabled=False."""
        panel = panel_and_mocks
        with patch("meetandread.config.set_config") as mock_set, \
             patch("meetandread.config.save_config") as mock_save:
            panel._on_waveform_toggled(0)
            mock_set.assert_called_once_with("ui.waveform_enabled", False)
            mock_save.assert_called_once()

    def test_handler_partial_state_persists_true(self, panel_and_mocks, manager):
        """Partially checked state (state=1) persists ui.waveform_enabled=True."""
        panel = panel_and_mocks
        with patch("meetandread.config.set_config") as mock_set, \
             patch("meetandread.config.save_config") as mock_save:
            panel._on_waveform_toggled(1)
            mock_set.assert_called_once_with("ui.waveform_enabled", True)
            mock_save.assert_called_once()

    def test_handler_does_not_crash_on_save_failure(self, panel_and_mocks, manager):
        """Handler swallows save failures without crashing."""
        panel = panel_and_mocks
        with patch("meetandread.config.set_config", side_effect=RuntimeError("disk full")):
            # Should not raise
            panel._on_waveform_toggled(0)

    def test_checkbox_exists(self, panel_and_mocks):
        """Panel has the waveform checkbox widget."""
        panel = panel_and_mocks
        assert hasattr(panel, '_waveform_checkbox')
        from PyQt6.QtWidgets import QCheckBox
        assert isinstance(panel._waveform_checkbox, QCheckBox)

    def test_checkbox_default_checked(self, panel_and_mocks):
        """Waveform checkbox defaults to checked."""
        panel = panel_and_mocks
        assert panel._waveform_checkbox.isChecked() is True

    def test_checkbox_object_name(self, panel_and_mocks):
        """Waveform checkbox uses AethericCheckBox object name."""
        panel = panel_and_mocks
        assert panel._waveform_checkbox.objectName() == "AethericCheckBox"

    def test_checkbox_label(self, panel_and_mocks):
        """Checkbox has the expected label text."""
        panel = panel_and_mocks
        assert "waveform" in panel._waveform_checkbox.text().lower()
        assert "recording" in panel._waveform_checkbox.text().lower()
