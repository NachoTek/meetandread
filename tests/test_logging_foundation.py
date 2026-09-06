"""Logging foundation tests (issue #98, docs/specs/issue-reporting.md).

Pure-logic, fast-lane compatible (ADR 0001): no ``windows`` marker needed.

Covers:
- Level normalization: normal runs at INFO, capture flag flips the same
  codebase to full DEBUG, no per-module override surface exists.
- Retention: normal-run logs older than 30 days are deleted, recent ones
  kept, capture-mode logs never deleted at any age.
- Log-file shape: one timestamped file per run; an INFO run excludes
  routine DEBUG chatter; a capture run includes it; the stdout tee stays
  console-mirroring only (transcript-bearing output never enters the log
  stream at INFO).
"""

import inspect
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from meetandread import logging_setup
from meetandread.logging_setup import (
    CAPTURE_LOG_PREFIX,
    NORMAL_LOG_PREFIX,
    cleanup_expired_logs,
    configure_logging,
    normalize_level,
    select_expired_normal_logs,
)


# Fixed clock so retention boundaries are deterministic.
NOW = datetime(2026, 9, 6, 12, 0, 0)


def _touch(path: Path, mtime: datetime) -> Path:
    """Create a small file with an explicit modification time."""
    path.write_text("log line\n", encoding="utf-8")
    ts = mtime.timestamp()
    os.utime(path, (ts, ts))
    return path


def _flush_root() -> None:
    for handler in logging.getLogger().handlers:
        handler.flush()


@pytest.fixture()
def isolated_logging():
    """Snapshot and restore root logger state and sys.stdout.

    configure_logging() replaces root handlers, the root level, and
    sys.stdout; tests must not leak any of that into the suite.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_stdout = sys.stdout
    try:
        yield root
    finally:
        sys.stdout = saved_stdout
        for handler in list(root.handlers):
            if handler not in saved_handlers:
                root.removeHandler(handler)
                try:
                    handler.close()
                except OSError:
                    pass
        root.setLevel(saved_level)


# ---------------------------------------------------------------------------
# Level normalization (all-or-nothing, root level)
# ---------------------------------------------------------------------------

class TestLevelNormalization:
    def test_normal_run_selects_info(self):
        assert normalize_level(capture_mode=False) == logging.INFO

    def test_capture_run_selects_debug(self):
        assert normalize_level(capture_mode=True) == logging.DEBUG

    def test_selection_is_all_or_nothing(self):
        """Only two possible levels — no gradient, no per-module mix."""
        for capture_mode in (False, True):
            assert normalize_level(capture_mode) in (logging.INFO, logging.DEBUG)

    def test_no_per_module_override_surface(self):
        """Level selection is a single boolean; no per-module knobs exist."""
        assert set(inspect.signature(normalize_level).parameters) == {"capture_mode"}
        assert set(inspect.signature(configure_logging).parameters) == {
            "logs_dir", "capture_mode", "now",
        }
        forbidden = [
            name
            for name in vars(logging_setup)
            if "module" in name.lower() and "level" in name.lower()
        ]
        assert forbidden == []

    def test_main_setup_logging_delegates_capture_flag(self, monkeypatch):
        """main.setup_logging is a thin hook forwarding the capture flag."""
        import meetandread.main as main_mod

        recorded = {}

        def fake_configure(**kwargs):
            recorded.update(kwargs)
            return Path("unused.log")

        monkeypatch.setattr(main_mod, "configure_logging", fake_configure)

        main_mod.setup_logging()
        assert recorded == {"capture_mode": False}

        main_mod.setup_logging(capture_mode=True)
        assert recorded == {"capture_mode": True}


# ---------------------------------------------------------------------------
# Retention — pure selector
# ---------------------------------------------------------------------------

class TestRetentionSelection:
    def test_normal_log_older_than_30_days_selected(self, tmp_path):
        old = _touch(
            tmp_path / "meetandread_20260101_120000.log", NOW - timedelta(days=31)
        )
        cutoff = NOW - timedelta(days=30)
        assert select_expired_normal_logs([old], cutoff) == [old]

    def test_recent_normal_log_not_selected(self, tmp_path):
        recent = _touch(
            tmp_path / "meetandread_20260901_120000.log", NOW - timedelta(days=29)
        )
        cutoff = NOW - timedelta(days=30)
        assert select_expired_normal_logs([recent], cutoff) == []

    def test_retention_boundary_strictly_older(self, tmp_path):
        cutoff = NOW - timedelta(days=30)
        just_newer = _touch(
            tmp_path / "meetandread_newer.log", cutoff + timedelta(seconds=1)
        )
        just_older = _touch(
            tmp_path / "meetandread_older.log", cutoff - timedelta(seconds=1)
        )
        assert select_expired_normal_logs([just_newer, just_older], cutoff) == [
            just_older
        ]

    def test_capture_log_never_selected_at_any_age(self, tmp_path):
        ancient_capture = _touch(
            tmp_path / f"{CAPTURE_LOG_PREFIX}20200101_120000.log",
            NOW - timedelta(days=365 * 5),
        )
        cutoff = NOW - timedelta(days=30)
        assert select_expired_normal_logs([ancient_capture], cutoff) == []

    def test_non_log_files_never_selected(self, tmp_path):
        stranger = _touch(
            tmp_path / "notes.txt", NOW - timedelta(days=400)
        )
        foreign_log = _touch(
            tmp_path / "someone_elses.log", NOW - timedelta(days=400)
        )
        cutoff = NOW - timedelta(days=30)
        assert select_expired_normal_logs([stranger, foreign_log], cutoff) == []


# ---------------------------------------------------------------------------
# Retention — startup cleanup
# ---------------------------------------------------------------------------

class TestRetentionCleanup:
    def test_deletes_old_normal_logs_keeps_recent(self, tmp_path):
        old = _touch(
            tmp_path / "meetandread_20260101_120000.log", NOW - timedelta(days=31)
        )
        recent = _touch(
            tmp_path / "meetandread_20260901_120000.log", NOW - timedelta(days=1)
        )

        deleted = cleanup_expired_logs(tmp_path, now=NOW)

        assert deleted == [old]
        assert not old.exists()
        assert recent.exists()

    def test_capture_logs_never_deleted_at_any_age(self, tmp_path):
        ancient_capture = _touch(
            tmp_path / f"{CAPTURE_LOG_PREFIX}20200101_120000.log",
            NOW - timedelta(days=365 * 5),
        )
        old_normal = _touch(
            tmp_path / "meetandread_20260101_120000.log", NOW - timedelta(days=45)
        )

        deleted = cleanup_expired_logs(tmp_path, now=NOW)

        assert deleted == [old_normal]
        assert ancient_capture.exists()
        assert not old_normal.exists()

    def test_capture_mode_artifacts_untouched(self, tmp_path):
        """Non-log files in the logs dir are never the cleanup's business."""
        artifact = _touch(
            tmp_path / "interaction_trace.jsonl", NOW - timedelta(days=400)
        )
        deleted = cleanup_expired_logs(tmp_path, now=NOW)
        assert deleted == []
        assert artifact.exists()

    def test_missing_logs_dir_is_noop(self, tmp_path):
        missing = tmp_path / "does-not-exist"
        assert cleanup_expired_logs(missing, now=NOW) == []

    def test_default_retention_window_is_30_days(self, tmp_path):
        boundary_old = _touch(
            tmp_path / "meetandread_old.log", datetime.now() - timedelta(days=31)
        )
        cleanup_expired_logs(tmp_path)
        assert not boundary_old.exists()


# ---------------------------------------------------------------------------
# configure_logging — file shape and level behavior
# ---------------------------------------------------------------------------

class TestConfigureLogging:
    def test_creates_timestamped_file_in_logs_dir(self, isolated_logging, tmp_path):
        log_file = configure_logging(logs_dir=tmp_path)
        assert log_file.parent == tmp_path
        assert log_file.name.startswith(NORMAL_LOG_PREFIX)
        assert log_file.name.endswith(".log")
        assert log_file.exists()

    def test_capture_run_uses_marked_filename(self, isolated_logging, tmp_path):
        log_file = configure_logging(logs_dir=tmp_path, capture_mode=True)
        assert log_file.name.startswith(CAPTURE_LOG_PREFIX)
        assert log_file.exists()

    def test_fixed_now_controls_timestamp(self, isolated_logging, tmp_path):
        log_file = configure_logging(
            logs_dir=tmp_path, now=datetime(2026, 9, 6, 8, 30, 5)
        )
        assert log_file.name == "meetandread_20260906_083005.log"

    def test_normal_run_excludes_routine_debug_chatter(
        self, isolated_logging, tmp_path
    ):
        log_file = configure_logging(logs_dir=tmp_path)
        module_logger = logging.getLogger("meetandread.transcription.worker")
        module_logger.debug("routine debug chatter from a deep module")
        module_logger.info("startup info record")
        _flush_root()

        content = log_file.read_text(encoding="utf-8")
        assert "startup info record" in content
        assert "routine debug chatter" not in content

    def test_capture_run_includes_debug_chatter_app_wide(
        self, isolated_logging, tmp_path
    ):
        log_file = configure_logging(logs_dir=tmp_path, capture_mode=True)
        for name in ("meetandread.audio.session", "meetandread.widgets.main_widget"):
            logging.getLogger(name).debug(f"chatter from {name}")
        _flush_root()

        content = log_file.read_text(encoding="utf-8")
        assert "chatter from meetandread.audio.session" in content
        assert "chatter from meetandread.widgets.main_widget" in content

    def test_root_level_set_to_selection(self, isolated_logging, tmp_path):
        configure_logging(logs_dir=tmp_path)
        assert logging.getLogger().level == logging.INFO
        configure_logging(logs_dir=tmp_path, capture_mode=True)
        assert logging.getLogger().level == logging.DEBUG


# ---------------------------------------------------------------------------
# Stdout tee — console-mirroring only
# ---------------------------------------------------------------------------

class TestStdoutTee:
    def test_transcript_bearing_stdout_never_enters_info_log(
        self, isolated_logging, tmp_path
    ):
        log_file = configure_logging(logs_dir=tmp_path)
        print("TRANSCRIPT-FRAGMENT canary text")
        _flush_root()

        content = log_file.read_text(encoding="utf-8")
        assert "TRANSCRIPT-FRAGMENT" not in content

    def test_console_mirroring_preserved(
        self, isolated_logging, tmp_path, capsys
    ):
        configure_logging(logs_dir=tmp_path)
        print("console visible line")
        assert "console visible line" in capsys.readouterr().out

    def test_reconfigure_does_not_duplicate_output(
        self, isolated_logging, tmp_path
    ):
        """A second configure call swaps the run handler; records land in
        the new file only, not duplicated into both."""
        first = configure_logging(logs_dir=tmp_path)
        second = configure_logging(logs_dir=tmp_path, capture_mode=True)
        logging.getLogger("meetandread.anything").info("one record")
        _flush_root()

        assert "one record" not in first.read_text(encoding="utf-8")
        assert "one record" in second.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Log-dir resolution seam (PR #115 storage-isolation pattern)
# ---------------------------------------------------------------------------

class TestLogsDirResolution:
    def test_resolve_logs_dir_delegates_to_storage_seam(self, monkeypatch, tmp_path):
        from meetandread.audio.storage import paths as storage_paths

        calls = {}

        def fake_get_logs_dir(base_dir=None):
            calls["base_dir"] = base_dir
            return tmp_path / "via-seam"

        monkeypatch.setattr(storage_paths, "get_logs_dir", fake_get_logs_dir)

        assert logging_setup.resolve_logs_dir() == tmp_path / "via-seam"
        logging_setup.resolve_logs_dir(base_dir=tmp_path / "home")
        assert calls["base_dir"] == tmp_path / "home"

    def test_default_resolution_lands_in_isolated_sandbox(
        self, isolated_logging, tmp_path
    ):
        """Default (no-arg) configure_logging must honor conftest isolation
        (tests/conftest.py, PR #115) — never the real ~/Documents tree."""
        log_file = configure_logging()
        assert log_file.is_relative_to(tmp_path)

    def test_configure_cleanup_scoped_to_given_dir(self, isolated_logging, tmp_path):
        """Retention inside configure_logging only touches the target dir."""
        doomed = _touch(
            tmp_path / "meetandread_20260101_120000.log", NOW - timedelta(days=60)
        )
        configure_logging(logs_dir=tmp_path, now=NOW)
        assert not doomed.exists()
