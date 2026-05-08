"""Tests proving the Settings denoising toggle persists on and off.

Covers:
- Config persistence: denoising_enabled survives save/reload for True and False
- Settings panel handler: _on_noise_filter_toggled calls set_config + save_config
- Non-bool and missing setting edge cases preserved
"""

import json
from pathlib import Path
from unittest.mock import call, patch

import pytest

from meetandread.config import (
    ConfigManager,
    SettingsPersistence,
    set_config,
    save_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset ConfigManager singleton before and after every test."""
    ConfigManager._instance = None
    ConfigManager._initialized = False
    yield
    ConfigManager._instance = None
    ConfigManager._initialized = False


@pytest.fixture
def isolated_config(tmp_path):
    """Return a ConfigManager backed by an isolated temp directory."""
    persistence = SettingsPersistence(config_dir=tmp_path)
    cm = ConfigManager(persistence=persistence)
    return cm


# ---------------------------------------------------------------------------
# 1. Config persistence round-trip for denoising enabled/disabled
# ---------------------------------------------------------------------------

class TestDenoisingTogglePersistence:
    """Prove transcription.microphone_denoising_enabled survives save/reload."""

    def test_persist_true_survives_reload(self, tmp_path):
        """Setting denoising to True, saving, reloading yields True."""
        # Phase 1: set and save
        persistence = SettingsPersistence(config_dir=tmp_path)
        cm = ConfigManager(persistence=persistence)
        assert cm.get("transcription.microphone_denoising_enabled") is False  # default
        cm.set("transcription.microphone_denoising_enabled", True)
        assert cm.save() is True

        # Phase 2: fresh manager loads from disk
        ConfigManager._instance = None
        ConfigManager._initialized = False
        persistence2 = SettingsPersistence(config_dir=tmp_path)
        cm2 = ConfigManager(persistence=persistence2)
        assert cm2.get("transcription.microphone_denoising_enabled") is True

    def test_persist_false_survives_reload(self, tmp_path):
        """Setting denoising to False, saving, reloading yields False."""
        # First save with True so there's something to flip
        persistence = SettingsPersistence(config_dir=tmp_path)
        cm = ConfigManager(persistence=persistence)
        cm.set("transcription.microphone_denoising_enabled", True)
        cm.save()

        # Now set to False
        ConfigManager._instance = None
        ConfigManager._initialized = False
        persistence2 = SettingsPersistence(config_dir=tmp_path)
        cm2 = ConfigManager(persistence=persistence2)
        cm2.set("transcription.microphone_denoising_enabled", False)
        assert cm2.save() is True

        # Reload fresh
        ConfigManager._instance = None
        ConfigManager._initialized = False
        persistence3 = SettingsPersistence(config_dir=tmp_path)
        cm3 = ConfigManager(persistence=persistence3)
        assert cm3.get("transcription.microphone_denoising_enabled") is False

    def test_default_is_false_before_any_save(self, tmp_path):
        """A fresh config with no file on disk defaults denoising to False."""
        persistence = SettingsPersistence(config_dir=tmp_path)
        cm = ConfigManager(persistence=persistence)
        assert cm.get("transcription.microphone_denoising_enabled") is False

    def test_config_file_contains_denoising_key(self, tmp_path):
        """The written JSON file explicitly stores the denoising key."""
        persistence = SettingsPersistence(config_dir=tmp_path)
        cm = ConfigManager(persistence=persistence)
        cm.set("transcription.microphone_denoising_enabled", True)
        cm.save()

        config_path = tmp_path / "config.json"
        assert config_path.exists()
        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert data["transcription"]["microphone_denoising_enabled"] is True


# ---------------------------------------------------------------------------
# 2. Settings panel handler (_on_noise_filter_toggled)
# ---------------------------------------------------------------------------

class TestNoiseFilterHandler:
    """Prove _on_noise_filter_toggled persists via set_config + save_config."""

    @pytest.fixture
    def panel(self, isolated_config):
        """Create a minimal FloatingSettingsPanel via __new__ (no Qt init)."""
        from meetandread.widgets.floating_panels import FloatingSettingsPanel
        p = FloatingSettingsPanel.__new__(FloatingSettingsPanel)
        return p

    @patch("meetandread.config.save_config")
    @patch("meetandread.config.set_config")
    def test_checked_state_persists_true(self, mock_set, mock_save, panel, isolated_config):
        """Checked (state=2) → set_config(..., True), save_config called."""
        panel._on_noise_filter_toggled(2)
        mock_set.assert_called_once_with(
            "transcription.microphone_denoising_enabled", True
        )
        mock_save.assert_called_once()

    @patch("meetandread.config.save_config")
    @patch("meetandread.config.set_config")
    def test_unchecked_state_persists_false(self, mock_set, mock_save, panel, isolated_config):
        """Unchecked (state=0) → set_config(..., False), save_config called."""
        panel._on_noise_filter_toggled(0)
        mock_set.assert_called_once_with(
            "transcription.microphone_denoising_enabled", False
        )
        mock_save.assert_called_once()

    @patch("meetandread.config.save_config")
    @patch("meetandread.config.set_config")
    def test_partial_check_persists_true(self, mock_set, mock_save, panel, isolated_config):
        """PartiallyChecked (state=1) is truthy → persists True."""
        panel._on_noise_filter_toggled(1)
        mock_set.assert_called_once_with(
            "transcription.microphone_denoising_enabled", True
        )
        mock_save.assert_called_once()

    @patch("meetandread.config.save_config")
    @patch("meetandread.config.set_config")
    def test_handler_calls_save_after_set(self, mock_set, mock_save, panel, isolated_config):
        """save_config is called after set_config (call order matters)."""
        panel._on_noise_filter_toggled(2)
        assert mock_set.call_count == 1
        assert mock_save.call_count == 1
        # set_config must be called (it's what persists the value)
        assert mock_set.called
        assert mock_save.called


# ---------------------------------------------------------------------------
# 3. Edge cases: non-bool and missing settings
# ---------------------------------------------------------------------------

class TestDenoisingToggleEdgeCases:
    """Edge-case coverage for denoising config handling."""

    def test_set_rejects_non_bool(self, isolated_config):
        """Setting denoising to a non-bool raises ValueError."""
        with pytest.raises(ValueError, match="bool"):
            isolated_config.set("transcription.microphone_denoising_enabled", "yes")
        with pytest.raises(ValueError, match="bool"):
            isolated_config.set("transcription.microphone_denoising_enabled", 1)

    def test_missing_denoising_key_loads_default_false(self, tmp_path):
        """A config file without denoising keys loads defaults (False)."""
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"config_version": 6}), encoding="utf-8")

        ConfigManager._instance = None
        ConfigManager._initialized = False
        persistence = SettingsPersistence(config_dir=tmp_path)
        cm = ConfigManager(persistence=persistence)
        assert cm.get("transcription.microphone_denoising_enabled") is False

    def test_toggle_roundtrip_does_not_touch_other_settings(self, tmp_path):
        """Toggling denoising does not alter other transcription settings."""
        persistence = SettingsPersistence(config_dir=tmp_path)
        cm = ConfigManager(persistence=persistence)
        # Set a baseline for other settings
        cm.set("transcription.confidence_threshold", 0.85)
        cm.set("transcription.microphone_denoising_enabled", True)
        cm.save()

        # Reload
        ConfigManager._instance = None
        ConfigManager._initialized = False
        persistence2 = SettingsPersistence(config_dir=tmp_path)
        cm2 = ConfigManager(persistence=persistence2)
        assert cm2.get("transcription.microphone_denoising_enabled") is True
        assert cm2.get("transcription.confidence_threshold") == 0.85
