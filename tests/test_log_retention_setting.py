"""User-selectable log retention tests (issue #112).

Fast-lane pure logic (ADR 0001): no ``windows`` marker needed.

Covers:
- Config key ``logging.log_retention_days``: default 30, persistence
  round-trip, malformed-value fallback, v8→v9 migration.
- Startup cleanup honors the configured value (shorter keeps more,
  longer deletes less; boundary is strictly-older).
- ``resolve_retention_days`` reads the config seam and falls back to
  30 on any failure.
- The named DEBUG event ``log_retention_cleanup`` (count deleted +
  retention value, safe fields only).
- The AC constraint: automatic cleanup NEVER deletes capture-mode
  DEBUG logs or capture artifacts, at ANY retention value.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from meetandread import logging_setup
from meetandread.config import (
    ConfigManager,
    SettingsPersistence,
)
from meetandread.logging_setup import (
    CAPTURE_LOG_PREFIX,
    LOG_RETENTION_DAYS,
    cleanup_expired_logs,
    resolve_retention_days,
)

# Fixed clock so retention boundaries are deterministic.
NOW = datetime(2026, 10, 1, 12, 0, 0)


def _touch(path: Path, mtime: datetime) -> Path:
    """Create a small file with an explicit modification time."""
    path.write_text("log line\n", encoding="utf-8")
    ts = mtime.timestamp()
    os.utime(path, (ts, ts))
    return path


@pytest.fixture(autouse=True)
def _reset_config_singleton():
    """Reset ConfigManager singleton before and after every test."""
    ConfigManager._instance = None
    ConfigManager._initialized = False
    yield
    ConfigManager._instance = None
    ConfigManager._initialized = False


@pytest.fixture
def isolated_config(tmp_path):
    """A ConfigManager backed by an isolated temp directory."""
    persistence = SettingsPersistence(config_dir=tmp_path)
    return ConfigManager(persistence=persistence)


# ---------------------------------------------------------------------------
# Config key: default, persistence, fallbacks, migration
# ---------------------------------------------------------------------------

class TestLogRetentionConfig:
    def test_default_is_30_days(self, isolated_config):
        assert (
            isolated_config.get("logging.log_retention_days")
            == LOG_RETENTION_DAYS
            == 30
        )

    def test_setting_survives_save_reload(self, tmp_path):
        persistence = SettingsPersistence(config_dir=tmp_path)
        cm = ConfigManager(persistence=persistence)
        cm.set("logging.log_retention_days", 7)
        assert cm.save() is True

        ConfigManager._instance = None
        ConfigManager._initialized = False
        persistence2 = SettingsPersistence(config_dir=tmp_path)
        cm2 = ConfigManager(persistence=persistence2)
        assert cm2.get("logging.log_retention_days") == 7

    def test_config_file_carries_logging_section(self, tmp_path):
        persistence = SettingsPersistence(config_dir=tmp_path)
        cm = ConfigManager(persistence=persistence)
        cm.set("logging.log_retention_days", 90)
        cm.save()
        data = json.loads(
            (tmp_path / "config.json").read_text(encoding="utf-8")
        )
        assert data["logging"]["log_retention_days"] == 90

    def test_malformed_values_fall_back_to_30(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "config_version": 9,
                    "logging": {"log_retention_days": "forever"},
                }
            ),
            encoding="utf-8",
        )
        persistence = SettingsPersistence(config_dir=tmp_path)
        cm = ConfigManager(persistence=persistence)
        assert cm.get("logging.log_retention_days") == 30

    @pytest.mark.parametrize("bad", [0, -5, 400, 3.5, True, None])
    def test_out_of_range_and_non_int_coerce_to_default(self, bad):
        from meetandread.config.models import LoggingSettings

        assert (
            LoggingSettings.from_dict({"log_retention_days": bad})
            .log_retention_days
            == 30
        )

    def test_migration_v8_adds_logging_section_with_default(self):
        persistence = SettingsPersistence()
        old = {
            "config_version": 8,
            "transcription": {"enabled": True},
        }
        migrated = persistence.migrate_config(old, 8)
        assert migrated["config_version"] == 9
        assert migrated["logging"]["log_retention_days"] == 30

    def test_migration_preserves_existing_value(self):
        persistence = SettingsPersistence()
        old = {
            "config_version": 8,
            "logging": {"log_retention_days": 60},
        }
        migrated = persistence.migrate_config(old, 8)
        assert migrated["logging"]["log_retention_days"] == 60

    def test_config_version_bumped_to_9(self):
        from meetandread.config.persistence import CURRENT_CONFIG_VERSION

        assert CURRENT_CONFIG_VERSION == 9


# ---------------------------------------------------------------------------
# resolve_retention_days (the config seam)
# ---------------------------------------------------------------------------

class TestResolveRetentionDays:
    def test_reads_configured_value(self, isolated_config):
        isolated_config.set("logging.log_retention_days", 14)
        assert resolve_retention_days() == 14

    def test_defaults_to_30_without_config(self, isolated_config):
        assert resolve_retention_days() == 30

    def test_falls_back_on_config_failure(self, monkeypatch):
        import meetandread.config.manager as cfg_manager

        def broken_get(key_path=None):
            raise RuntimeError("config unavailable")

        monkeypatch.setattr(
            cfg_manager.ConfigManager, "get", broken_get
        )
        assert resolve_retention_days() == 30


# ---------------------------------------------------------------------------
# Cleanup honors the setting
# ---------------------------------------------------------------------------

class TestCleanupHonorsRetention:
    def test_shorter_retention_deletes_more(self, tmp_path):
        ten_day_old = _touch(
            tmp_path / "meetandread_20260101_120000.log",
            NOW - timedelta(days=10),
        )
        assert cleanup_expired_logs(
            tmp_path, now=NOW, retention_days=30
        ) == []
        assert ten_day_old.exists()
        assert cleanup_expired_logs(
            tmp_path, now=NOW, retention_days=7
        ) == [ten_day_old]
        assert not ten_day_old.exists()

    def test_longer_retention_keeps_more(self, tmp_path):
        forty_day_old = _touch(
            tmp_path / "meetandread_20260101_120000.log",
            NOW - timedelta(days=40),
        )
        assert cleanup_expired_logs(
            tmp_path, now=NOW, retention_days=30
        ) == [forty_day_old]
        # Recreate; 90-day retention keeps the 40-day-old log.
        forty_day_old = _touch(
            tmp_path / "meetandread_20260101_120000.log",
            NOW - timedelta(days=40),
        )
        assert cleanup_expired_logs(
            tmp_path, now=NOW, retention_days=90
        ) == []
        assert forty_day_old.exists()

    def test_boundary_strictly_older_at_custom_retention(self, tmp_path):
        cutoff = NOW - timedelta(days=7)
        just_older = _touch(
            tmp_path / "meetandread_20260101_120000.log",
            cutoff - timedelta(seconds=1),
        )
        just_newer = _touch(
            tmp_path / "meetandread_20260102_120000.log",
            cutoff + timedelta(seconds=1),
        )
        deleted = cleanup_expired_logs(
            tmp_path, now=NOW, retention_days=7
        )
        assert deleted == [just_older]
        assert just_newer.exists()

    def test_config_value_flows_through_default_resolution(
        self, tmp_path, isolated_config
    ):
        """cleanup with no explicit retention resolves the config seam."""
        isolated_config.set("logging.log_retention_days", 3)
        four_day_old = _touch(
            tmp_path / "meetandread_20260101_120000.log",
            NOW - timedelta(days=4),
        )
        two_day_old = _touch(
            tmp_path / "meetandread_20260102_120000.log",
            NOW - timedelta(days=2),
        )
        deleted = cleanup_expired_logs(tmp_path, now=NOW)
        assert deleted == [four_day_old]
        assert two_day_old.exists()


# ---------------------------------------------------------------------------
# The named DEBUG event (issue #112 named-event convention)
# ---------------------------------------------------------------------------

class TestCleanupNamedEvent:
    def test_event_emitted_with_count_and_retention(
        self, tmp_path, caplog
    ):
        _touch(
            tmp_path / "meetandread_20260101_120000.log",
            NOW - timedelta(days=40),
        )
        _touch(
            tmp_path / "meetandread_20260102_120000.log",
            NOW - timedelta(days=45),
        )
        with caplog.at_level(logging.DEBUG, logger="meetandread.logging_setup"):
            cleanup_expired_logs(tmp_path, now=NOW, retention_days=30)
        debugs = [
            r.getMessage()
            for r in caplog.records
            if r.name == "meetandread.logging_setup"
            and r.levelno == logging.DEBUG
        ]
        assert (
            "log_retention_cleanup: deleted=2 retention_days=30" in debugs
        ), debugs

    def test_event_quiet_at_info(self, tmp_path, caplog):
        _touch(
            tmp_path / "meetandread_20260101_120000.log",
            NOW - timedelta(days=40),
        )
        with caplog.at_level(logging.INFO, logger="meetandread.logging_setup"):
            cleanup_expired_logs(tmp_path, now=NOW, retention_days=30)
        loud = [
            r.getMessage()
            for r in caplog.records
            if r.name == "meetandread.logging_setup"
            and r.levelno >= logging.INFO
        ]
        assert loud == [], loud


# ---------------------------------------------------------------------------
# AC: automatic cleanup never deletes capture artifacts, at ANY
# retention value (proving the exclusion survives the new setting)
# ---------------------------------------------------------------------------

class TestCaptureExclusionAtAnyRetention:
    def test_capture_logs_and_artifacts_survive_even_one_day_retention(
        self, tmp_path
    ):
        """The strongest form of the AC: retention_days=1, capture
        artifacts years old — nothing capture-related is deleted."""
        capture_log = _touch(
            tmp_path / f"{CAPTURE_LOG_PREFIX}20200101_120000.log",
            NOW - timedelta(days=365 * 5),
        )
        trace = _touch(
            tmp_path / "interaction_trace.jsonl",
            NOW - timedelta(days=365 * 5),
        )
        marker = _touch(
            tmp_path / "capture_complete.marker",
            NOW - timedelta(days=365 * 5),
        )
        bundle = _touch(
            tmp_path / "diagnostics_bundle.txt",
            NOW - timedelta(days=365 * 5),
        )
        expired_normal = _touch(
            tmp_path / "meetandread_20260101_120000.log",
            NOW - timedelta(days=40),
        )

        deleted = cleanup_expired_logs(
            tmp_path, now=NOW, retention_days=1
        )

        # Only the expired NORMAL log went; the scan ran for real.
        assert deleted == [expired_normal]
        assert not expired_normal.exists()
        for survivor in (capture_log, trace, marker, bundle):
            assert survivor.exists(), survivor

    def test_retention_scan_never_enters_capture_directory(self, tmp_path):
        """A capture directory (the on-disk home of capture artifacts
        since #104) sits BESIDE the logs dir; its contents — including
        a normal-SHAPED name — are never touched, at any retention."""
        from meetandread.capture_mode import COMPLETION_MARKER_NAME

        normal_logs = tmp_path / "normal-logs"
        normal_logs.mkdir()
        capture_dir = tmp_path / "captures" / "run-20200101_120000"
        capture_dir.mkdir(parents=True)

        expired_normal = _touch(
            normal_logs / "meetandread_20200101_120000.log",
            NOW - timedelta(days=365 * 5),
        )
        normal_shaped_in_capture = _touch(
            capture_dir / "meetandread_20200101_120000.log",
            NOW - timedelta(days=365 * 5),
        )
        capture_log = _touch(
            capture_dir / f"{CAPTURE_LOG_PREFIX}20200101_120000.log",
            NOW - timedelta(days=365 * 5),
        )
        marker = _touch(
            capture_dir / COMPLETION_MARKER_NAME,
            NOW - timedelta(days=365 * 5),
        )

        deleted = cleanup_expired_logs(
            normal_logs, now=NOW, retention_days=1
        )

        assert deleted == [expired_normal]
        assert not expired_normal.exists()
        for survivor in (
            normal_shaped_in_capture,
            capture_log,
            marker,
        ):
            assert survivor.exists(), survivor
