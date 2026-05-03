"""Tests for benchmark_history field in TranscriptionSettings and config migration v1→v2.

Covers:
- Default empty benchmark_history on new TranscriptionSettings
- Round-trip serialization (to_dict / from_dict)
- Graceful handling of missing key in from_dict
- Config migration from v1 to v2 (adds benchmark_history)
- Direct loading of v2 config with existing benchmark_history entries
"""

import json
import tempfile
from pathlib import Path

import pytest

from meetandread.config.models import AppSettings, TranscriptionSettings
from meetandread.config.persistence import SettingsPersistence


# ---------------------------------------------------------------------------
# TranscriptionSettings model tests
# ---------------------------------------------------------------------------

def test_benchmark_history_default_empty():
    """New TranscriptionSettings should have an empty benchmark_history dict."""
    settings = TranscriptionSettings()
    assert settings.benchmark_history == {}
    assert isinstance(settings.benchmark_history, dict)


def test_benchmark_history_round_trip():
    """to_dict() followed by from_dict() should preserve benchmark_history entries."""
    original = TranscriptionSettings(
        benchmark_history={
            "base": {"wer": 17.3, "timestamp": "2026-04-26T18:00:00"},
            "small": {"wer": 12.1, "timestamp": "2026-04-26T19:00:00"},
        }
    )
    data = original.to_dict()
    restored = TranscriptionSettings.from_dict(data)

    assert restored.benchmark_history == original.benchmark_history
    assert restored.benchmark_history["base"]["wer"] == 17.3
    assert restored.benchmark_history["small"]["timestamp"] == "2026-04-26T19:00:00"


def test_from_dict_missing_benchmark_history():
    """from_dict() with no benchmark_history key should yield empty dict."""
    data = {"enabled": True, "realtime_model_size": "tiny"}
    settings = TranscriptionSettings.from_dict(data)

    assert settings.benchmark_history == {}
    assert settings.enabled is True


# ---------------------------------------------------------------------------
# Config persistence / migration tests
# ---------------------------------------------------------------------------

def test_config_v1_migrates_to_v2(tmp_path):
    """Loading a v1 config should migrate to current version with empty benchmark_history."""
    v1_config = {
        "config_version": 1,
        "model": {"realtime_model_size": "auto"},
        "transcription": {
            "enabled": True,
            "realtime_model_size": "tiny",
            "postprocess_model_size": "base",
        },
        "hardware": {},
        "ui": {},
    }

    # Write v1 config to disk
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(v1_config), encoding="utf-8")

    persistence = SettingsPersistence(config_dir=tmp_path)
    settings = persistence.load_settings()

    # Should be migrated to current version
    from meetandread.config.persistence import CURRENT_CONFIG_VERSION
    assert settings.config_version == CURRENT_CONFIG_VERSION
    # benchmark_history should exist and be empty
    assert settings.transcription.benchmark_history == {}
    # Existing values should be preserved
    assert settings.transcription.realtime_model_size == "tiny"
    assert settings.transcription.postprocess_model_size == "base"


def test_config_v2_loads_directly(tmp_path):
    """Loading a v2 config with benchmark_history entries should preserve them."""
    v2_config = {
        "config_version": 2,
        "model": {"realtime_model_size": "auto"},
        "transcription": {
            "enabled": True,
            "realtime_model_size": "tiny",
            "postprocess_model_size": "base",
            "enable_postprocessing": True,
            "benchmark_history": {
                "base": {"wer": 17.35, "timestamp": "2026-04-26T19:30:00"},
                "tiny": {"wer": 25.0, "timestamp": "2026-04-26T19:35:00"},
            },
        },
        "hardware": {},
        "ui": {},
    }

    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(v2_config), encoding="utf-8")

    persistence = SettingsPersistence(config_dir=tmp_path)
    settings = persistence.load_settings()

    # Migrated to current version, benchmark_history preserved
    from meetandread.config.persistence import CURRENT_CONFIG_VERSION
    assert settings.config_version == CURRENT_CONFIG_VERSION
    assert settings.transcription.benchmark_history["base"]["wer"] == 17.35
    assert settings.transcription.benchmark_history["tiny"]["wer"] == 25.0
    assert settings.transcription.benchmark_history["base"]["timestamp"] == "2026-04-26T19:30:00"
