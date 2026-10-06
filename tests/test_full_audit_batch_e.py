"""Full-audit batch E logging tests (issue #103).

Final batch of the full-audit DEBUG pass: recording controller and
management, configuration persistence and models, the performance
package (benchmark, resource monitor), the hardware detector and model
recommender, utilities, the Feature Dependency checker, the
single-instance guard, and the application entry module. After this
batch, zero unaudited modules remain.

Per module: (a) DEBUG-trail tests for named internal events (queue and
preemption decisions, config load/save, benchmark runs, dependency
check outcomes, startup sequence), (b) INFO quietness (per-step events
must not appear at INFO — an INFO-level run shows a readable
operational summary with no per-event DEBUG spam), (c) operational
facts asserted at INFO. Mirrors the caplog prior art of batches
A/B/C/D (tests/test_full_audit_batch_{a,b,c,d}.py).

Capture-boundary privacy (amended spec, Logging + Privacy sections):
DEBUG and INFO events are semantic only — counts, durations, model
names, stage transitions, sanitized statuses — and never carry
Transcript text, recording titles, speaker names, or filesystem paths
into the log stream. Privacy canaries drive distinctive text through
each seam and assert it never appears in the trail at any level
(formatted output included).

Runs at the pure-logic seam (spec: docs/specs/issue-reporting.md —
"existing seams reused, no new ones"); the authoritative pass runs
under the Windows venv (ADR 0001).
"""

import json
import logging
import sys
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from meetandread.config.manager import ConfigManager, validate_storage_paths
from meetandread.config.models import AppSettings, StoragePaths
from meetandread.config.persistence import SettingsPersistence
from meetandread.dependencies import (
    FeatureDependency,
    check_feature_dependencies,
    is_dependency_available,
    reset_availability_cache,
)
from meetandread.hardware.detector import HardwareDetector, SystemSpecs
from meetandread.hardware.recommender import (
    ModelRecommender,
    recommend_model_size,
)
from meetandread.performance.benchmark import BenchmarkRunner
from meetandread.performance.monitor import ResourceMonitor, ResourceSnapshot
from meetandread.recording.cleanup_queue import CleanupQueue
from meetandread.recording.controller import RecordingController
from meetandread.recording.management import (
    delete_recording_structured,
    enumerate_recording_files,
    rename_recording,
)


# ---------------------------------------------------------------------------
# caplog helpers (batch B/C/D prior art)
# ---------------------------------------------------------------------------


def _debug(caplog, logger_name: str) -> list:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == logger_name and r.levelno == logging.DEBUG
    ]


def _info(caplog, logger_name: str) -> list:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == logger_name and r.levelno == logging.INFO
    ]


def _at_or_above_info(caplog, logger_name: str) -> list:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == logger_name and r.levelno >= logging.INFO
    ]


def _formatted_output(caplog, logger_name: str) -> list:
    """Fully formatted log lines — message PLUS appended exception text.

    Privacy canaries must inspect this, not just ``getMessage()``: a
    record logged with ``exc_info`` has the raw exception message and
    traceback appended by the formatter, which getMessage() alone never
    shows (batch C prior art).
    """
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    return [
        formatter.format(r)
        for r in caplog.records
        if r.name == logger_name
    ]


def _starts_with(messages: list, prefix: str) -> list:
    return [m for m in messages if m.startswith(prefix)]


# ---------------------------------------------------------------------------
# Privacy canaries
# ---------------------------------------------------------------------------

CANARY_TRANSCRIPT = "CANARY_TranscriptTextSecret"
CANARY_SPEAKER_NAME = "CanarySpeakerName"
CANARY_STEM = "canary-secret-board-meeting"
CANARY_PATH_FRAGMENT = "canary-user-home"

PERSIST_LOG = "meetandread.config.persistence"
MANAGER_LOG = "meetandread.config.manager"
DEPS_LOG = "meetandread.dependencies"
HWDET_LOG = "meetandread.hardware.detector"
HWREC_LOG = "meetandread.hardware.recommender"
BENCH_LOG = "meetandread.performance.benchmark"
MON_LOG = "meetandread.performance.monitor"
CTRL_LOG = "meetandread.recording.controller"
MGMT_LOG = "meetandread.recording.management"
CLEANUP_LOG = "meetandread.recording.cleanup_queue"
MAIN_LOG = "meetandread.main"


def _make_specs(
    total_ram_gb: float = 16.0,
    cpu_count_logical: int = 8,
) -> SystemSpecs:
    return SystemSpecs(
        total_ram_gb=total_ram_gb,
        available_ram_gb=total_ram_gb / 2,
        cpu_count_logical=cpu_count_logical,
        cpu_count_physical=max(1, cpu_count_logical // 2),
        cpu_freq_mhz=2400.0,
        is_64bit=True,
        platform="Windows",
    )


def _make_wav(path: Path, n_frames: int = 100) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * n_frames)
    return path


@pytest.fixture()
def fresh_config_manager(monkeypatch: pytest.MonkeyPatch):
    """Reset the ConfigManager singleton around each manager test."""
    monkeypatch.setattr(ConfigManager, "_instance", None)
    monkeypatch.setattr(ConfigManager, "_initialized", False)
    yield
    monkeypatch.setattr(ConfigManager, "_instance", None)
    monkeypatch.setattr(ConfigManager, "_initialized", False)


@pytest.fixture()
def _fresh_dependency_cache():
    reset_availability_cache()
    yield
    reset_availability_cache()# ===========================================================================
# config/persistence.py
# ===========================================================================


class TestPersistenceLogging:
    def test_first_load_uses_defaults_info_summary(self, tmp_path, caplog) -> None:
        persistence = SettingsPersistence(config_dir=tmp_path)
        with caplog.at_level(logging.INFO, logger=PERSIST_LOG):
            persistence.load_settings()
        infos = _info(caplog, PERSIST_LOG)
        assert _starts_with(infos, "config_defaults_used:")

    def test_load_emits_debug_trail(self, tmp_path, caplog) -> None:
        persistence = SettingsPersistence(config_dir=tmp_path)
        persistence.get_config_path().write_text("{}", encoding="utf-8")
        with caplog.at_level(logging.DEBUG, logger=PERSIST_LOG):
            persistence.load_settings()
        debugs = _debug(caplog, PERSIST_LOG)
        assert _starts_with(debugs, "config_loaded:")

    def test_migration_named_event(self, tmp_path, caplog) -> None:
        persistence = SettingsPersistence(config_dir=tmp_path)
        persistence.get_config_path().write_text(
            json.dumps({"config_version": 1}), encoding="utf-8"
        )
        with caplog.at_level(logging.INFO, logger=PERSIST_LOG):
            persistence.load_settings()
        infos = _info(caplog, PERSIST_LOG)
        assert _starts_with(infos, "config_migrated:")

    def test_save_emits_debug_event_without_path(self, tmp_path, caplog) -> None:
        persistence = SettingsPersistence(config_dir=tmp_path / CANARY_STEM)
        with caplog.at_level(logging.DEBUG, logger=PERSIST_LOG):
            assert persistence.save_settings(AppSettings.get_defaults())
        saves = _starts_with(_debug(caplog, PERSIST_LOG), "config_saved:")
        assert saves
        assert CANARY_STEM not in saves[0]

    def test_corrupt_config_named_warning(self, tmp_path, caplog) -> None:
        persistence = SettingsPersistence(config_dir=tmp_path)
        persistence.get_config_path().write_text("{not json", encoding="utf-8")
        with caplog.at_level(logging.DEBUG, logger=PERSIST_LOG):
            persistence.load_raw()
        above = _at_or_above_info(caplog, PERSIST_LOG)
        assert any(m.startswith("config_load_failed:") for m in above)

    def test_canary_path_never_logged(self, tmp_path, caplog) -> None:
        config_dir = tmp_path / CANARY_PATH_FRAGMENT / CANARY_STEM
        persistence = SettingsPersistence(config_dir=config_dir)
        with caplog.at_level(logging.DEBUG, logger=PERSIST_LOG):
            persistence.load_settings()
            persistence.save_settings(AppSettings.get_defaults())
        for line in _formatted_output(caplog, PERSIST_LOG):
            assert CANARY_PATH_FRAGMENT not in line
            assert CANARY_STEM not in line


# ===========================================================================
# config/manager.py
# ===========================================================================


class TestConfigManagerLogging:
    def test_init_info_summary_without_path(
        self, tmp_path, caplog, fresh_config_manager
    ) -> None:
        persistence = SettingsPersistence(config_dir=tmp_path)
        with caplog.at_level(logging.INFO, logger=MANAGER_LOG):
            ConfigManager(persistence=persistence)
        infos = _info(caplog, MANAGER_LOG)
        assert _starts_with(infos, "config_ready:")
        assert str(tmp_path) not in infos[0]

    def test_set_emits_named_debug_event(
        self, tmp_path, caplog, fresh_config_manager
    ) -> None:
        persistence = SettingsPersistence(config_dir=tmp_path)
        manager = ConfigManager(persistence=persistence)
        with caplog.at_level(logging.DEBUG, logger=MANAGER_LOG):
            manager.set("transcription.confidence_threshold", 0.8)
        debugs = _debug(caplog, MANAGER_LOG)
        assert _starts_with(debugs, "config_value_set:")

    def test_save_info_summary(
        self, tmp_path, caplog, fresh_config_manager
    ) -> None:
        persistence = SettingsPersistence(config_dir=tmp_path)
        manager = ConfigManager(persistence=persistence)
        manager.set("transcription.confidence_threshold", 0.8)
        with caplog.at_level(logging.INFO, logger=MANAGER_LOG):
            manager.save()
        infos = _info(caplog, MANAGER_LOG)
        assert _starts_with(infos, "config_saved:")

    def test_storage_path_validation_failure_named_event(
        self, tmp_path, caplog
    ) -> None:
        blocked = tmp_path / "blocked"
        blocked.mkdir()
        real_resolve = Path.resolve

        def _resolve_fail(self, *args, **kwargs):
            if self == blocked:
                raise RuntimeError("resolve boom")
            return real_resolve(self, *args, **kwargs)

        paths = StoragePaths(transcripts_path=str(blocked))
        with patch.object(Path, "resolve", _resolve_fail):
            with caplog.at_level(logging.DEBUG, logger=MANAGER_LOG):
                errors = validate_storage_paths(paths)
        assert errors
        assert any(
            m.startswith("storage_path_validation_failed:")
            for m in _at_or_above_info(caplog, MANAGER_LOG)
        )


# ===========================================================================
# dependencies.py
# ===========================================================================


class TestDependenciesLogging:
    def setup_method(self):
        reset_availability_cache()

    def teardown_method(self):
        reset_availability_cache()

    def test_available_probe_named_debug_event(self, caplog) -> None:
        dep = FeatureDependency(
            name="testdep", module="json", feature="testing",
            resolution_dev="dev", resolution_frozen="frozen",
        )
        with caplog.at_level(logging.DEBUG, logger=DEPS_LOG):
            assert is_dependency_available(dep)
        debugs = _debug(caplog, DEPS_LOG)
        assert _starts_with(debugs, "dependency_available:")

    def test_unavailable_probe_named_info_event(self, caplog) -> None:
        dep = FeatureDependency(
            name="missingdep", module="definitely_not_a_module_xyz",
            feature="testing",
            resolution_dev="dev", resolution_frozen="frozen",
        )
        with caplog.at_level(logging.INFO, logger=DEPS_LOG):
            assert not is_dependency_available(dep)
        infos = _info(caplog, DEPS_LOG)
        assert _starts_with(infos, "dependency_unavailable:")
        assert all("ImportError" not in m or "error_class=" in m for m in infos)

    def test_check_all_summary(self, caplog) -> None:
        with caplog.at_level(logging.INFO, logger=DEPS_LOG):
            check_feature_dependencies()
        infos = _info(caplog, DEPS_LOG)
        assert any(m.startswith("dependency_check_complete:") for m in infos)

    def test_unavailable_error_class_only(self, caplog) -> None:
        dep = FeatureDependency(
            name="missingdep", module="definitely_not_a_module_xyz",
            feature="testing",
            resolution_dev="dev", resolution_frozen="frozen",
        )
        with caplog.at_level(logging.DEBUG, logger=DEPS_LOG):
            is_dependency_available(dep)
        for line in _formatted_output(caplog, DEPS_LOG):
            assert "No module named" not in line


# ===========================================================================
# hardware/detector.py + hardware/recommender.py
# ===========================================================================


class TestHardwareLogging:
    def test_detect_named_debug_and_info_summary(
        self, tmp_path, caplog, monkeypatch
    ) -> None:
        detector = HardwareDetector(cache_ttl_seconds=60)
        monkeypatch.setattr(
            "meetandread.hardware.detector.psutil.virtual_memory",
            lambda: SimpleNamespace(total=16 * 1024**3, available=8 * 1024**3),
        )
        monkeypatch.setattr(
            "meetandread.hardware.detector.psutil.cpu_count",
            lambda logical=True: 8 if logical else 4,
        )
        monkeypatch.setattr(
            "meetandread.hardware.detector.psutil.cpu_freq",
            lambda: SimpleNamespace(current=2400.0),
        )
        with caplog.at_level(logging.DEBUG, logger=HWDET_LOG):
            detector.detect()
        debugs = _debug(caplog, HWDET_LOG)
        assert _starts_with(debugs, "hardware_detected:")
        assert _starts_with(_info(caplog, HWDET_LOG), "hardware_specs_ready:")

    def test_cache_hit_debug_event(self, caplog, monkeypatch) -> None:
        detector = HardwareDetector(cache_ttl_seconds=60)
        monkeypatch.setattr(
            "meetandread.hardware.detector.psutil.virtual_memory",
            lambda: SimpleNamespace(total=16 * 1024**3, available=8 * 1024**3),
        )
        monkeypatch.setattr(
            "meetandread.hardware.detector.psutil.cpu_count",
            lambda logical=True: 8 if logical else 4,
        )
        monkeypatch.setattr(
            "meetandread.hardware.detector.psutil.cpu_freq",
            lambda: SimpleNamespace(current=2400.0),
        )
        detector.detect()
        with caplog.at_level(logging.DEBUG, logger=HWDET_LOG):
            detector.detect()  # second call hits the cache
        assert _starts_with(_debug(caplog, HWDET_LOG), "hardware_cache_hit:")

    def test_requirements_check_debug_event(self, caplog) -> None:
        detector = HardwareDetector()
        with caplog.at_level(logging.DEBUG, logger=HWDET_LOG):
            detector.has_minimum_requirements(_make_specs(), dual_mode=True)
        assert _starts_with(
            _debug(caplog, HWDET_LOG), "hardware_requirements_checked:"
        )

    def test_recommender_events(self, caplog) -> None:
        detector = MagicMock(spec=HardwareDetector)
        detector.refresh.return_value = _make_specs()
        recommender = ModelRecommender(hardware_detector=detector)
        with caplog.at_level(logging.DEBUG, logger=HWREC_LOG):
            recommender.detect_and_recommend()
        debugs = _debug(caplog, HWREC_LOG)
        assert _starts_with(debugs, "model_recommendation_computed:")
        assert _starts_with(
            _info(caplog, HWREC_LOG), "model_recommendation_ready:"
        )

    def test_recommend_model_size_pure(self) -> None:
        assert recommend_model_size(_make_specs(total_ram_gb=4, cpu_count_logical=2)) == "tiny"
        assert recommend_model_size(_make_specs()) == "tiny"

    def test_save_failure_warning_no_exception_payload(self, caplog) -> None:
        detector = MagicMock(spec=HardwareDetector)
        detector.refresh.return_value = _make_specs()
        recommender = ModelRecommender(hardware_detector=detector)
        recommender.detect_and_recommend()
        with patch(
            "meetandread.config.set_config",
            side_effect=RuntimeError("config store exploded"),
        ):
            with caplog.at_level(logging.DEBUG, logger=HWREC_LOG):
                assert not recommender.save_recommendation_to_config()
        warnings = [
            m for m in _at_or_above_info(caplog, HWREC_LOG)
            if m.startswith("recommendation_save_failed:")
        ]
        assert warnings
        assert "config store exploded" not in warnings[0]
        assert "error_class=RuntimeError" in warnings[0]


# ===========================================================================
# performance/benchmark.py
# ===========================================================================


def _make_benchmark_engine(tmp_path: Path, text: str = "hello world"):
    engine = MagicMock()
    engine.is_model_loaded.return_value = True
    seg = SimpleNamespace(text=text)
    engine.transcribe_chunk.return_value = [seg]
    engine.get_model_info.return_value = {"size": "tiny"}
    return engine


class TestBenchmarkLogging:
    def _make_runner(self, tmp_path: Path, engine=None, **kwargs) -> BenchmarkRunner:
        from meetandread.performance.benchmark import BenchmarkRunner

        clip = _make_wav(tmp_path / "benchmark.wav", n_frames=16000)
        truth = tmp_path / "ground_truth.txt"
        truth.write_text("hello world", encoding="utf-8")
        return BenchmarkRunner(
            engine=engine or _make_benchmark_engine(tmp_path),
            test_clip_path=clip,
            ground_truth_path=truth,
            **kwargs,
        )

    def test_run_named_debug_trail(self, tmp_path, caplog) -> None:
        runner = self._make_runner(tmp_path)
        with caplog.at_level(logging.DEBUG, logger=BENCH_LOG):
            runner.run()
        debugs = _debug(caplog, BENCH_LOG)
        assert _starts_with(debugs, "benchmark_audio_loaded:")
        assert _starts_with(debugs, "benchmark_chunk_done:")

    def test_run_info_summary(self, tmp_path, caplog) -> None:
        runner = self._make_runner(tmp_path)
        with caplog.at_level(logging.INFO, logger=BENCH_LOG):
            runner.run()
        infos = _info(caplog, BENCH_LOG)
        assert _starts_with(infos, "benchmark_started:")
        assert _starts_with(infos, "benchmark_complete:")

    def test_info_quietness(self, tmp_path, caplog) -> None:
        runner = self._make_runner(tmp_path)
        with caplog.at_level(logging.INFO, logger=BENCH_LOG):
            runner.run()
        msgs = [r.getMessage() for r in caplog.records if r.name == BENCH_LOG]
        assert not _starts_with(msgs, "benchmark_chunk_done:")

    def test_canary_transcript_never_logged(self, tmp_path, caplog) -> None:
        from meetandread.performance.benchmark import BenchmarkRunner

        clip = _make_wav(tmp_path / "benchmark.wav", n_frames=16000)
        truth = tmp_path / "ground_truth.txt"
        truth.write_text(CANARY_TRANSCRIPT, encoding="utf-8")
        engine = _make_benchmark_engine(tmp_path, text=CANARY_TRANSCRIPT)
        runner = BenchmarkRunner(
            engine=engine, test_clip_path=clip, ground_truth_path=truth,
        )
        with caplog.at_level(logging.DEBUG, logger=BENCH_LOG):
            runner.run()
        for line in _formatted_output(caplog, BENCH_LOG):
            assert CANARY_TRANSCRIPT not in line

    def test_cancel_named_event(self, tmp_path, caplog) -> None:
        engine = _make_benchmark_engine(tmp_path)
        real_transcribe = engine.transcribe_chunk
        seen = {"n": 0}

        def chunk_spy(chunk_audio, **kwargs):
            seen["n"] += 1
            if seen["n"] >= 1:
                runner._cancel_event.set()
            return real_transcribe(chunk_audio, **kwargs)

        engine.transcribe_chunk = chunk_spy
        runner = self._make_runner(tmp_path, engine=engine)
        runner._chunk_duration_s = 0.4  # 1s clip spans multiple chunks
        with caplog.at_level(logging.INFO, logger=BENCH_LOG):
            runner.run()
        above = _at_or_above_info(caplog, BENCH_LOG)
        assert any(m.startswith("benchmark_cancelled:") for m in above)


# ===========================================================================
# performance/monitor.py
# ===========================================================================


class TestResourceMonitorLogging:
    def test_poll_named_debug_event(self, caplog) -> None:
        monitor = ResourceMonitor()
        with patch(
            "meetandread.performance.monitor.psutil.virtual_memory",
            lambda: SimpleNamespace(
                percent=42.0, available=8 * 1024**3, total=16 * 1024**3
            ),
        ), patch(
            "meetandread.performance.monitor.psutil.cpu_percent",
            lambda interval=None: 11.0,
        ):
            with caplog.at_level(logging.DEBUG, logger=MON_LOG):
                monitor.poll()
        assert _starts_with(_debug(caplog, MON_LOG), "resource_snapshot:")

    def test_threshold_warning_named_event(self, caplog) -> None:
        monitor = ResourceMonitor(ram_warning_percent=50.0)
        with patch(
            "meetandread.performance.monitor.psutil.virtual_memory",
            lambda: SimpleNamespace(
                percent=91.0, available=1 * 1024**3, total=16 * 1024**3
            ),
        ), patch(
            "meetandread.performance.monitor.psutil.cpu_percent",
            lambda interval=None: 11.0,
        ):
            with caplog.at_level(logging.DEBUG, logger=MON_LOG):
                monitor.poll()
        above = _at_or_above_info(caplog, MON_LOG)
        assert any(m.startswith("resource_threshold_exceeded:") for m in above)

    def test_info_quietness(self, caplog) -> None:
        monitor = ResourceMonitor()
        with patch(
            "meetandread.performance.monitor.psutil.virtual_memory",
            lambda: SimpleNamespace(
                percent=42.0, available=8 * 1024**3, total=16 * 1024**3
            ),
        ), patch(
            "meetandread.performance.monitor.psutil.cpu_percent",
            lambda interval=None: 11.0,
        ):
            with caplog.at_level(logging.INFO, logger=MON_LOG):
                monitor.poll()
        msgs = [r.getMessage() for r in caplog.records if r.name == MON_LOG]
        assert not _starts_with(msgs, "resource_snapshot:")


# ===========================================================================
# recording/management.py
# ===========================================================================


class TestRecordingManagementLogging:
    def test_enumerate_named_debug_event(self, tmp_path, caplog) -> None:
        rec = tmp_path / "recordings"
        tra = tmp_path / "transcripts"
        rec.mkdir()
        tra.mkdir()
        _make_wav(rec / f"{CANARY_STEM}.wav")
        with caplog.at_level(logging.DEBUG, logger=MGMT_LOG):
            found = enumerate_recording_files(
                CANARY_STEM, recordings_dir=rec, transcripts_dir=tra
            )
        assert found
        events = _starts_with(_debug(caplog, MGMT_LOG), "files_enumerated:")
        assert events
        assert "count=" in events[0]
        assert CANARY_STEM not in events[0]

    def test_rename_info_summary_without_stem(self, tmp_path, caplog) -> None:
        rec = tmp_path / "recordings"
        tra = tmp_path / "transcripts"
        rec.mkdir()
        tra.mkdir()
        _make_wav(rec / f"{CANARY_STEM}.wav")
        with caplog.at_level(logging.INFO, logger=MGMT_LOG):
            result = rename_recording(
                CANARY_STEM, "renamed-stem",
                recordings_dir=rec, transcripts_dir=tra,
            )
        assert result.renamed
        summaries = _starts_with(_info(caplog, MGMT_LOG), "recording_renamed:")
        assert summaries
        assert CANARY_STEM not in summaries[0]

    def test_rename_conflict_named_warning(self, tmp_path, caplog) -> None:
        rec = tmp_path / "recordings"
        tra = tmp_path / "transcripts"
        rec.mkdir()
        tra.mkdir()
        _make_wav(rec / f"{CANARY_STEM}.wav")
        _make_wav(rec / "renamed-stem.wav")
        with caplog.at_level(logging.DEBUG, logger=MGMT_LOG):
            result = rename_recording(
                CANARY_STEM, "renamed-stem",
                recordings_dir=rec, transcripts_dir=tra,
            )
        assert result.failed
        above = _at_or_above_info(caplog, MGMT_LOG)
        assert any(m.startswith("rename_conflict:") for m in above)

    def test_delete_info_summary(self, tmp_path, caplog) -> None:
        rec = tmp_path / "recordings"
        tra = tmp_path / "transcripts"
        rec.mkdir()
        tra.mkdir()
        _make_wav(rec / f"{CANARY_STEM}.wav")
        with caplog.at_level(logging.INFO, logger=MGMT_LOG):
            result = delete_recording_structured(
                CANARY_STEM, recordings_dir=rec, transcripts_dir=tra
            )
        assert result.all_succeeded
        summaries = _starts_with(_info(caplog, MGMT_LOG), "recording_deleted:")
        assert summaries
        assert CANARY_STEM not in summaries[0]

    def test_canary_stem_never_logged(self, tmp_path, caplog) -> None:
        rec = tmp_path / "recordings"
        tra = tmp_path / "transcripts"
        rec.mkdir()
        tra.mkdir()
        _make_wav(rec / f"{CANARY_STEM}.wav")
        (tra / f"{CANARY_STEM}.md").write_text("# t", encoding="utf-8")
        with caplog.at_level(logging.DEBUG, logger=MGMT_LOG):
            enumerate_recording_files(
                CANARY_STEM, recordings_dir=rec, transcripts_dir=tra
            )
            rename_recording(
                CANARY_STEM, "renamed-stem",
                recordings_dir=rec, transcripts_dir=tra,
            )
            delete_recording_structured(
                "renamed-stem", recordings_dir=rec, transcripts_dir=tra
            )
        for line in _formatted_output(caplog, MGMT_LOG):
            assert CANARY_STEM not in line


# ===========================================================================
# recording/cleanup_queue.py
# ===========================================================================


class TestCleanupQueueLogging:
    def test_enqueue_named_debug_event(self, tmp_path, caplog) -> None:
        queue = CleanupQueue(
            queue_path=tmp_path / "queue.json",
            recordings_dir=tmp_path / "recordings",
            transcripts_dir=tmp_path / "transcripts",
        )
        with caplog.at_level(logging.DEBUG, logger=CLEANUP_LOG):
            queue.enqueue_file_deletion(CANARY_STEM)
        events = _starts_with(_debug(caplog, CLEANUP_LOG), "cleanup_enqueued:")
        assert events
        assert CANARY_STEM not in events[0]

    def test_process_info_summary(self, tmp_path, caplog) -> None:
        rec = tmp_path / "recordings"
        rec.mkdir(parents=True)
        _make_wav(rec / f"{CANARY_STEM}.wav")
        queue = CleanupQueue(
            queue_path=tmp_path / "queue.json",
            recordings_dir=rec,
            transcripts_dir=tmp_path / "transcripts",
        )
        queue.enqueue_file_deletion(CANARY_STEM)
        with caplog.at_level(logging.INFO, logger=CLEANUP_LOG):
            result = queue.process_pending()
        assert result.processed == 1
        summaries = _starts_with(_info(caplog, CLEANUP_LOG), "cleanup_processed:")
        assert summaries
        assert CANARY_STEM not in summaries[0]

    def test_corrupt_queue_named_warning(self, tmp_path, caplog) -> None:
        queue_file = tmp_path / "queue.json"
        queue_file.write_text("{corrupt", encoding="utf-8")
        with caplog.at_level(logging.DEBUG, logger=CLEANUP_LOG):
            CleanupQueue(
                queue_path=queue_file,
                recordings_dir=tmp_path / "recordings",
                transcripts_dir=tmp_path / "transcripts",
            )
        above = _at_or_above_info(caplog, CLEANUP_LOG)
        assert any(m.startswith("cleanup_queue_reset:") for m in above)

    def test_load_debug_event(self, tmp_path, caplog) -> None:
        queue_file = tmp_path / "queue.json"
        queue_file.parent.mkdir(parents=True, exist_ok=True)
        queue_file.write_text(
            json.dumps({"operations": [
                {"kind": "file_delete", "target": "some-stem",
                 "paths": [], "status": "pending", "attempts": 0},
            ]}),
            encoding="utf-8",
        )
        with caplog.at_level(logging.DEBUG, logger=CLEANUP_LOG):
            CleanupQueue(
                queue_path=queue_file,
                recordings_dir=tmp_path / "recordings",
                transcripts_dir=tmp_path / "transcripts",
            )
        assert _starts_with(_debug(caplog, CLEANUP_LOG), "cleanup_queue_loaded:")

    def test_canary_stem_never_logged(self, tmp_path, caplog) -> None:
        rec = tmp_path / "recordings"
        rec.mkdir(parents=True)
        _make_wav(rec / f"{CANARY_STEM}.wav")
        queue = CleanupQueue(
            queue_path=tmp_path / "queue.json",
            recordings_dir=rec,
            transcripts_dir=tmp_path / "transcripts",
        )
        with caplog.at_level(logging.DEBUG, logger=CLEANUP_LOG):
            queue.enqueue_file_deletion(CANARY_STEM)
            queue.process_pending()
        for line in _formatted_output(caplog, CLEANUP_LOG):
            assert CANARY_STEM not in line


# ===========================================================================
# recording/controller.py — queue lifecycle + preemption decisions
# ===========================================================================


class TestControllerQueueLogging:
    def _make_controller(self) -> RecordingController:
        controller = RecordingController(enable_transcription=False)
        controller._post_processor = MagicMock()
        controller._post_processor.preempt_current_job.return_value = True
        return controller

    def test_preempt_decision_named_event(self, caplog) -> None:
        controller = self._make_controller()
        with caplog.at_level(logging.INFO, logger=CTRL_LOG):
            controller.preempt_post_processing(reason="new recording starting")
        infos = _info(caplog, CTRL_LOG)
        assert _starts_with(infos, "post_processing_preempted:")

    def test_preempt_none_no_info_event(self, caplog) -> None:
        controller = self._make_controller()
        controller._post_processor.preempt_current_job.return_value = False
        with caplog.at_level(logging.INFO, logger=CTRL_LOG):
            controller.preempt_post_processing(reason="new recording starting")
        infos = _info(caplog, CTRL_LOG)
        assert not _starts_with(infos, "post_processing_preempted:")

    def test_retry_decision_events(self, tmp_path, caplog) -> None:
        controller = self._make_controller()
        transcript = tmp_path / f"{CANARY_STEM}.md"
        transcript.write_text("# t", encoding="utf-8")
        controller._post_processor.get_status_for_audio.return_value = None
        job = SimpleNamespace(job_id="job123")
        controller._post_processor.schedule_post_process.return_value = job
        with patch(
            "meetandread.audio.storage.paths.get_recordings_dir",
            return_value=tmp_path,
        ):
            _make_wav(tmp_path / f"{CANARY_STEM}.wav")
            with caplog.at_level(logging.INFO, logger=CTRL_LOG):
                controller.retry_post_processing(transcript)
        infos = _info(caplog, CTRL_LOG)
        assert _starts_with(infos, "post_processing_retry_scheduled:")
        for line in _formatted_output(caplog, CTRL_LOG):
            assert CANARY_STEM not in line

    def test_retry_missing_audio_named_warning(self, tmp_path, caplog) -> None:
        controller = self._make_controller()
        transcript = tmp_path / f"{CANARY_STEM}.md"
        transcript.write_text("# t", encoding="utf-8")
        controller._post_processor.get_status_for_audio.return_value = None
        with patch(
            "meetandread.audio.storage.paths.get_recordings_dir",
            return_value=tmp_path,
        ):
            with caplog.at_level(logging.DEBUG, logger=CTRL_LOG):
                assert controller.retry_post_processing(transcript) is None
        above = _at_or_above_info(caplog, CTRL_LOG)
        warnings = [m for m in above if m.startswith("retry_unavailable_audio:")]
        assert warnings
        assert CANARY_STEM not in warnings[0]

    def test_hotplug_snapshot_no_friendly_names_at_info(
        self, caplog
    ) -> None:
        from meetandread.audio import SourceConfig
        from meetandread.recording.controller import ActiveSourceIdentity

        controller = self._make_controller()
        configs = [SourceConfig(type="mic", device_id=1)]

        # Simulate an identity carrying a user-visible device name.
        def identity_with_canary_name(config):
            return ActiveSourceIdentity(
                type=config.type,
                device_id=str(config.device_id),
                friendly_name=CANARY_SPEAKER_NAME,
                flow="capture",
            )

        controller._source_identity_from_config = identity_with_canary_name
        with caplog.at_level(logging.INFO, logger=CTRL_LOG):
            controller._snapshot_active_sources(configs)
        for line in _formatted_output(caplog, CTRL_LOG):
            assert CANARY_SPEAKER_NAME not in line

    def test_diagnostics_unavailable_stays_debug(self, caplog) -> None:
        controller = self._make_controller()
        controller._session = MagicMock()
        controller._session.get_stats.side_effect = RuntimeError("boom")
        with caplog.at_level(logging.INFO, logger=CTRL_LOG):
            controller.get_diagnostics()
        msgs = [r.getMessage() for r in caplog.records if r.name == CTRL_LOG]
        assert not _starts_with(msgs, "diagnostics_unavailable:")


# ===========================================================================
# dependencies banner path in main.py (check_critical_dlls / startup steps)
# ===========================================================================


class TestMainStartupLogging:
    def test_dll_check_failure_named_event(self, caplog, monkeypatch) -> None:
        import meetandread.main as main_mod

        monkeypatch.setattr(sys, "frozen", True, raising=False)
        with patch.dict(
            # A None entry makes __import__ raise ImportError for the lib.
            sys.modules, {"pywhispercpp": None},
        ), patch.object(
            main_mod.QMessageBox, "critical", MagicMock()
        ), patch.object(
            sys, "exit", side_effect=SystemExit
        ):
            with caplog.at_level(logging.DEBUG, logger=MAIN_LOG):
                with pytest.raises(SystemExit):
                    main_mod.check_critical_dlls()
        above = _at_or_above_info(caplog, MAIN_LOG)
        failures = [m for m in above if m.startswith("critical_dll_check_failed:")]
        assert failures
        assert "error_class=" in failures[0]
