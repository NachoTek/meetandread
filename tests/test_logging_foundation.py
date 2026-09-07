"""Logging foundation tests (issue #98, docs/specs/issue-reporting.md).

Pure-logic, fast-lane compatible (ADR 0001): no ``windows`` marker needed.

Covers:
- Level normalization: normal runs at INFO, capture flag flips the same
  codebase to full DEBUG, no per-module override surface exists.
- Retention: normal-run logs older than 30 days are deleted, recent ones
  kept, capture-mode logs never deleted at any age.
- Log-file shape: one timestamped file per run; an INFO run excludes
  routine DEBUG chatter; a capture run includes it; the stdout tee stays
  console-mirroring only (transcript-bearing output never enters the
  log stream).
- Transcript-leak canaries: the two ``_on_phrase_result`` handlers run
  their real code paths under capture-mode DEBUG with a distinctive
  transcript text; the capture log must contain the telemetry fields
  but never the transcript text.
"""

import inspect
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

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
from meetandread.capture_mode import (
    COMPLETION_MARKER_NAME,
    configure_capture_logging,
    read_completion_marker,
    write_completion_marker,
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
            "logs_dir", "capture_mode", "now", "is_capture_dir", "handler_cls",
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

    def test_main_setup_logging_capture_dir_routes_to_capture_mode(
        self, isolated_logging, tmp_path
    ):
        """A capture dir selects the capture-directory logging path
        (issue #104) — full DEBUG into the capture dir, prompt-flushed."""
        import meetandread.main as main_mod

        log_file = main_mod.setup_logging(capture_dir=tmp_path / "cap")
        assert log_file.parent == tmp_path / "cap"
        assert log_file.name.startswith(CAPTURE_LOG_PREFIX)
        assert logging.getLogger().level == logging.DEBUG


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
            tmp_path / "meetandread_20260807_120001.log",
            cutoff + timedelta(seconds=1),
        )
        just_older = _touch(
            tmp_path / "meetandread_20260807_115959.log",
            cutoff - timedelta(seconds=1),
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

    def test_wrong_shape_names_never_selected(self, tmp_path):
        """Only the exact meetandread_YYYYMMDD_HHMMSS.log shape is a
        candidate — foreign meetandread-prefixed artifacts survive."""
        cutoff = NOW - timedelta(days=30)
        wrong_shape_names = [
            "meetandread_notes.log",
            "meetandread_20260101.log",
            "meetandread_20260101_1200.log",
            "meetandread_2026AB0101_120000.log",
            "meetandread_99999999_999999.log",
        ]
        wrong_shape = [
            _touch(tmp_path / name, NOW - timedelta(days=400))
            for name in wrong_shape_names
        ]
        assert select_expired_normal_logs(wrong_shape, cutoff) == []


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
            tmp_path / "meetandread_20260101_120000.log",
            datetime.now() - timedelta(days=31),
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

    def test_transcript_bearing_stdout_never_enters_capture_log(
        self, isolated_logging, tmp_path
    ):
        """Capture mode runs at DEBUG, yet stdout still stays console-only."""
        log_file = configure_logging(logs_dir=tmp_path, capture_mode=True)
        logging.getLogger("meetandread.capture.probe").debug("capture probe line")
        print("TRANSCRIPT-FRAGMENT capture canary")
        _flush_root()

        content = log_file.read_text(encoding="utf-8")
        assert "capture probe line" in content
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
# Transcript-leak canaries — real handlers under capture-mode DEBUG
# ---------------------------------------------------------------------------

class TestTranscriptLeakCanaries:
    """PR #116 review finding: capture DEBUG must never record transcript.

    The two ``_on_phrase_result`` handlers are the seam where segment
    text previously flowed into a ``logger.debug`` call. These canaries
    run the REAL methods (unbound, with a minimal fake ``self``) under
    capture-mode DEBUG on a temp storage dir and assert the distinctive
    transcript canary never reaches the capture log file — while the
    non-content telemetry (conf/final/idx/...) still does, proving the
    handler body actually executed.
    """

    CANARY = "CANARY_TRANSCRIPT_LEAK_xyzzy"

    def _make_canary_result(self):
        from meetandread.transcription.accumulating_processor import (
            SegmentResult,
        )

        return SegmentResult(
            text=self.CANARY,
            confidence=87,
            start_time=1.0,
            end_time=2.0,
            segment_index=3,
            is_final=True,
            phrase_start=True,
        )

    def _capture_log_content(self, tmp_path):
        _flush_root()
        logs = sorted(tmp_path.glob("meetandread_capture_*.log"))
        assert logs, "capture log file was not created"
        return logs[-1].read_text(encoding="utf-8")

    def test_controller_phrase_handler_never_logs_transcript(
        self, isolated_logging, tmp_path
    ):
        from meetandread.recording.controller import RecordingController

        configure_logging(logs_dir=tmp_path, capture_mode=True)

        word_count_calls = []

        controller_self = SimpleNamespace(
            _try_live_speaker_match=lambda: None,
            _segment_to_words=(
                lambda result: RecordingController._segment_to_words(None, result)
            ),
            _transcript_store=SimpleNamespace(
                get_word_count=lambda: 0,
                commit_live_phrase=lambda: word_count_calls.append("commit"),
                set_live_phrase_words=lambda words: word_count_calls.append("set"),
            ),
            on_phrase_result=None,
        )

        RecordingController._on_phrase_result(
            controller_self, self._make_canary_result()
        )

        content = self._capture_log_content(tmp_path)
        assert self.CANARY not in content
        # Handler ran: telemetry present and word-count path exercised.
        assert "Segment received" in content
        assert "conf: 87" in content
        assert word_count_calls, "word-count path was not exercised"

    def test_main_widget_phrase_handler_never_logs_transcript(
        self, isolated_logging, tmp_path
    ):
        from meetandread.widgets.main_widget import MeetAndReadWidget

        configure_logging(logs_dir=tmp_path, capture_mode=True)

        widget_self = SimpleNamespace(
            _cc_overlay=None,
        )

        MeetAndReadWidget._on_phrase_result(
            widget_self, self._make_canary_result()
        )

        content = self._capture_log_content(tmp_path)
        assert self.CANARY not in content
        # Handler ran: telemetry present without the transcript text.
        assert "Segment received" in content
        assert "conf: 87" in content
        assert "phrase_start: True" in content


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

    def test_configure_capture_dir_skips_retention(self, isolated_logging, tmp_path):
        """Retention never runs against a capture directory (issue #104):
        an expired normal-shaped log inside a capture dir survives a
        configure_logging call made with is_capture_dir=True."""
        from meetandread.logging_setup import CAPTURE_LOG_PREFIX as _prefix

        capture_dir = tmp_path / "capture-home"
        capture_dir.mkdir()
        ancient = _touch(
            capture_dir / "meetandread_20200101_120000.log",
            NOW - timedelta(days=365 * 5),
        )
        log_file = configure_logging(
            logs_dir=capture_dir,
            capture_mode=True,
            now=NOW,
            is_capture_dir=True,
        )
        assert log_file.parent == capture_dir
        assert log_file.name.startswith(_prefix)
        assert ancient.exists()


# ---------------------------------------------------------------------------
# Capture-directory contract (issue #104) — pure fast-lane coverage
# ---------------------------------------------------------------------------

class TestCaptureDirectoryContract:
    """The artifact edge of the one new seam (docs/specs/issue-reporting.md).

    Pure fast-lane coverage of the capture-directory contract: flag
    parsing, capture-dir logging layout, retention exclusion, prompt
    flush on write, marker write/read round-trip, and the torn-record
    reader. Subprocess launch/kill coverage lives in the windows-marked
    test_capture_mode_subprocess.py (ADR 0001).
    """

    def test_flag_absent_means_normal_run(self):
        from meetandread.capture_mode import parse_capture_flag

        assert parse_capture_flag([]) is None

    def test_flag_parses_directory(self, tmp_path):
        from meetandread.capture_mode import parse_capture_flag

        parsed = parse_capture_flag(["--issue-capture", str(tmp_path / "cap")])
        assert parsed == tmp_path / "cap"

    def test_flag_rejects_existing_file(self, tmp_path):
        from meetandread.capture_mode import CaptureModeError, parse_capture_flag

        existing = tmp_path / "file.txt"
        existing.write_text("not a dir", encoding="utf-8")
        with pytest.raises(CaptureModeError):
            parse_capture_flag(["--issue-capture", str(existing)])

    def test_flag_rejects_non_empty_existing_directory(self, tmp_path):
        """One capture directory == one run (fix round 1): a directory
        that exists and holds ANYTHING is rejected at entry — a stale
        completion marker must never make a killed reused run read as
        a clean exit."""
        from meetandread.capture_mode import CaptureModeError, parse_capture_flag

        reused = tmp_path / "reused-capture"
        reused.mkdir()
        (reused / COMPLETION_MARKER_NAME).write_text(
            '{"finished_at": "2026-09-06T10:00:00"}', encoding="utf-8"
        )
        with pytest.raises(CaptureModeError, match="not empty"):
            parse_capture_flag(["--issue-capture", str(reused)])

    def test_flag_accepts_existing_empty_directory(self, tmp_path):
        from meetandread.capture_mode import parse_capture_flag

        empty = tmp_path / "fresh-capture"
        empty.mkdir()
        assert parse_capture_flag(["--issue-capture", str(empty)]) == empty

    def test_flag_accepts_missing_directory(self, tmp_path):
        from meetandread.capture_mode import parse_capture_flag

        assert parse_capture_flag(["--issue-capture", str(tmp_path / "nope")]) == (
            tmp_path / "nope"
        )

    def test_unknown_flags_ignored_by_capture_parse(self, tmp_path):
        """The capture parser must not choke on the app's other flags."""
        from meetandread.capture_mode import parse_capture_flag

        parsed = parse_capture_flag(
            ["--some-future-flag", "value", "--issue-capture", str(tmp_path)]
        )
        assert parsed == tmp_path

    def test_configure_capture_logging_streams_debug_into_capture_dir(
        self, isolated_logging, tmp_path
    ):
        capture_dir = tmp_path / "cap" / "run-1"  # nested: created if missing
        log_file = configure_capture_logging(
            capture_dir, now=datetime(2026, 9, 6, 9, 30, 0)
        )
        assert capture_dir.is_dir()
        assert log_file.parent == capture_dir
        assert log_file.name == "meetandread_capture_20260906_093000.log"

        logging.getLogger("meetandread.anything.deep").debug(
            "capture chatter line"
        )
        content = log_file.read_text(encoding="utf-8")
        assert "capture chatter line" in content
        assert logging.getLogger().level == logging.DEBUG

    def test_capture_log_prompt_flushed_without_explicit_flush(
        self, isolated_logging, tmp_path
    ):
        """Prompt-flush contract: a completed record is on disk
        immediately after its emit returns — no explicit flush needed."""
        capture_dir = tmp_path / "cap"
        log_file = configure_capture_logging(capture_dir)

        logging.getLogger("meetandread.flush.probe").info(
            "immediately-on-disk probe"
        )

        # No _flush_root() call: the prompt-flush handler must already
        # have written the record to the file (flush+fsync per emit).
        content = log_file.read_text(encoding="utf-8")
        assert "immediately-on-disk probe" in content

    def test_completion_marker_round_trip(self, tmp_path):
        capture_dir = tmp_path / "cap"
        capture_dir.mkdir()
        marker = write_completion_marker(
            capture_dir, now=datetime(2026, 9, 6, 10, 0, 0)
        )
        assert marker.name == COMPLETION_MARKER_NAME
        data = read_completion_marker(capture_dir)
        assert data is not None
        assert data["finished_at"] == "2026-09-06T10:00:00"

    def test_missing_marker_reads_none(self, tmp_path):
        assert read_completion_marker(tmp_path) is None

    def test_retention_never_selects_capture_dir_contents(self, tmp_path):
        """AC6: nothing inside a capture directory is ever a retention
        candidate — even a file whose name matches the normal-run shape
        exactly, at any age. The retention scan runs against the normal
        logs dir (where it deletes an expired normal log, proving the
        scan really ran) while the capture dir sits untouched beside it."""
        normal_logs = tmp_path / "normal-logs"
        normal_logs.mkdir()
        capture_dir = tmp_path / "issue-capture-2026-09-06"
        capture_dir.mkdir()

        expired_normal = _touch(
            normal_logs / "meetandread_20200101_120000.log",
            NOW - timedelta(days=365 * 5),
        )
        # A capture dir can contain normal-SHAPED names (e.g. a copied
        # log) — they must still never be selected, because the scan
        # never enters a capture directory.
        normal_shaped_in_capture = _touch(
            capture_dir / "meetandread_20200101_120000.log",
            NOW - timedelta(days=365 * 5),
        )
        capture_log = _touch(
            capture_dir / f"{CAPTURE_LOG_PREFIX}20200101_120000.log",
            NOW - timedelta(days=365 * 5),
        )
        marker = _touch(
            capture_dir / COMPLETION_MARKER_NAME, NOW - timedelta(days=400)
        )

        deleted = cleanup_expired_logs(normal_logs, now=NOW)

        # The scan ran and did its normal job...
        assert deleted == [expired_normal]
        assert not expired_normal.exists()
        # ...and nothing inside the capture directory was touched.
        assert normal_shaped_in_capture.exists()
        assert capture_log.exists()
        assert marker.exists()

    def test_read_appendable_records_tolerates_torn_tail(self, tmp_path):
        from meetandread.capture_mode import read_appendable_records

        log = tmp_path / "capture.log"
        log.write_text(
            "line one\nline two\nline three torn, no newline",
            encoding="utf-8",
        )
        assert read_appendable_records(log) == ["line one", "line two"]

    def test_read_appendable_records_complete_file(self, tmp_path):
        from meetandread.capture_mode import read_appendable_records

        log = tmp_path / "capture.log"
        log.write_text("a\nb\nc\n", encoding="utf-8")
        assert read_appendable_records(log) == ["a", "b", "c"]

    def test_read_appendable_records_missing_file(self, tmp_path):
        from meetandread.capture_mode import read_appendable_records

        assert read_appendable_records(tmp_path / "nope.log") == []

    def test_read_appendable_records_keeps_blank_interior_lines(self, tmp_path):
        from meetandread.capture_mode import read_appendable_records

        log = tmp_path / "capture.log"
        log.write_text("a\n\nb\n", encoding="utf-8")
        assert read_appendable_records(log) == ["a", "", "b"]

    def test_same_second_second_start_fails_loudly_never_truncates(
        self, isolated_logging, tmp_path
    ):
        """Fix round 1, collision half of one-dir-one-run: the capture
        log file is created EXCLUSIVELY (O_CREAT|O_EXCL). A second run
        starting in the same second resolves to the same filename and
        must FAIL — the first run's stream is never truncated."""
        from meetandread.capture_mode import (
            CaptureModeError,
            configure_capture_logging,
        )

        capture_dir = tmp_path / "cap"
        fixed_now = datetime(2026, 9, 6, 11, 22, 33)
        first = configure_capture_logging(capture_dir, now=fixed_now)
        logging.getLogger("meetandread.first.run").info("first run record")
        # NOTE: the first handler stays installed; simulate the second
        # start's configure attempt directly (a second PROCESS would do
        # exactly this against the same dir + filename).

        # The first run's log file must be intact after the failed start.
        with pytest.raises(CaptureModeError, match="already exists"):
            configure_capture_logging(capture_dir, now=fixed_now)

        content = first.read_text(encoding="utf-8")
        assert "first run record" in content
        # No half-written second file, no truncation: exactly one log.
        logs = list(capture_dir.glob(f"{CAPTURE_LOG_PREFIX}*.log"))
        assert [p.name for p in logs] == [first.name]

    def test_capture_logging_configured_seam(self, isolated_logging, tmp_path):
        """The bootstrap/main idempotence seam: capture_logging_configured()
        flips True exactly when a prompt-flush run handler is installed."""
        from meetandread.capture_mode import (
            capture_logging_configured,
            configure_capture_logging,
        )

        assert capture_logging_configured() is False
        configure_capture_logging(tmp_path / "cap")
        assert capture_logging_configured() is True

    def test_capture_logging_configured_false_for_plain_handler(
        self, isolated_logging, tmp_path
    ):
        """A normal-run FileHandler is NOT the capture handler — the
        seam must not confuse the two."""
        from meetandread.capture_mode import capture_logging_configured

        configure_logging(logs_dir=tmp_path)
        assert capture_logging_configured() is False
