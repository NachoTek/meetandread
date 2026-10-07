"""Tests proving the Settings log-retention control persists (issue #112).

Covers:
- The Settings panel handler persists via set_config + save_config
  with the ``logging.log_retention_days`` key.
- The named DEBUG toggle event ``log_retention_set`` (privacy-safe
  fields only, #119 convention).
"""

import logging
from unittest.mock import patch

import pytest

from meetandread.config import (
    ConfigManager,
    SettingsPersistence,
)

FP_LOG = "meetandread.widgets.floating_panels"


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
    return ConfigManager(persistence=persistence)


class TestLogRetentionHandler:
    @pytest.fixture
    def panel(self, isolated_config):
        """Create a minimal FloatingSettingsPanel via __new__ (no Qt init)."""
        from meetandread.widgets.floating_panels import FloatingSettingsPanel
        return FloatingSettingsPanel.__new__(FloatingSettingsPanel)

    @patch("meetandread.config.save_config")
    @patch("meetandread.config.set_config")
    def test_value_persists(self, mock_set, mock_save, panel, isolated_config):
        panel._on_log_retention_changed(14)
        mock_set.assert_called_once_with("logging.log_retention_days", 14)
        mock_save.assert_called_once()

    @patch("meetandread.config.save_config")
    @patch("meetandread.config.set_config")
    def test_other_value_persists(self, mock_set, mock_save, panel, isolated_config):
        panel._on_log_retention_changed(90)
        mock_set.assert_called_once_with("logging.log_retention_days", 90)
        mock_save.assert_called_once()


class TestLogRetentionDebugEvents:
    @pytest.fixture
    def panel(self, isolated_config):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel
        return FloatingSettingsPanel.__new__(FloatingSettingsPanel)

    def test_change_emits_named_debug_event(self, panel, isolated_config, caplog):
        with caplog.at_level(logging.DEBUG, logger=FP_LOG):
            panel._on_log_retention_changed(14)
        debugs = [
            r.getMessage()
            for r in caplog.records
            if r.name == FP_LOG and r.levelno == logging.DEBUG
        ]
        assert "log_retention_set: days=14 source=handler" in debugs, debugs

    def test_event_quiet_at_info(self, panel, isolated_config, caplog):
        with caplog.at_level(logging.INFO, logger=FP_LOG):
            panel._on_log_retention_changed(14)
        loud = [
            r.getMessage()
            for r in caplog.records
            if r.name == FP_LOG and r.levelno >= logging.INFO
        ]
        assert loud == [], loud
