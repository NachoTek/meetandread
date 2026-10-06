"""Environment info artifact tests (issue #108,
docs/specs/issue-reporting.md).

The environment info — app version, OS, hardware class — is a bundle
constituent (spec "Captured content"). The writer lives on the APP side
of the capture-directory seam (written at capture start, into the
capture directory); the bundle assembler reads it back.

Pure-logic, fast-lane (ADR 0001): the module under test is stdlib-only
— importing it must not pull the hardware detector (which imports
psutil), because the bootstrap writes the initial artifact BEFORE the
app subsystems are imported. ``hardware_class_from_specs`` accepts the
raw figures and derives the class without importing anything heavy.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from meetandread.environment_info import (
    ENVIRONMENT_FILE_NAME,
    classify_hardware_class,
    read_environment_info,
    write_environment_info,
)

# ---------------------------------------------------------------------------
# The defensive boundary: stdlib-only module
# ---------------------------------------------------------------------------

_HEAVY_TOP_LEVELS = {
    "PyQt6",
    "sounddevice",
    "pyaudiowpatch",
    "sherpa_onnx",
    "numpy",
    "scipy",
    "pywhispercpp",
    "comtypes",
    "webrtcvad",
    "psutil",
}


class TestDefensiveBoundary:
    def test_importing_environment_info_pulls_no_heavy_stack(self):
        import os as _os

        code = (
            "import sys, meetandread.environment_info; "
            "heavy = sorted(m for m in sys.modules "
            "if m.split('.')[0] in {"
            + ",".join(repr(m) for m in sorted(_HEAVY_TOP_LEVELS))
            + "}); "
            "print(heavy)"
        )
        env = dict(_os.environ)
        env["PYTHONPATH"] = str(
            Path(__import__("meetandread.environment_info", fromlist=["x"])
                 .__file__)
            .resolve()
            .parent.parent
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "[]", (
            f"environment_info imports heavy modules: {out.stdout.strip()}"
        )


# ---------------------------------------------------------------------------
# Hardware class (pure function of the raw figures)
# ---------------------------------------------------------------------------


class TestHardwareClass:
    def test_high_class(self):
        # The dual-mode bar (>= 8 GB AND >= 4 cores) IS "high": a
        # machine meeting it exactly belongs to the high class.
        assert (
            classify_hardware_class(
                total_ram_gb=16.0, cpu_count_logical=12
            )
            == "high"
        )
        assert (
            classify_hardware_class(
                total_ram_gb=8.0, cpu_count_logical=4
            )
            == "high"
        )

    def test_medium_class(self):
        # Meets the single-mode bar but not the dual-mode one.
        assert (
            classify_hardware_class(
                total_ram_gb=4.0, cpu_count_logical=2
            )
            == "medium"
        )
        assert (
            classify_hardware_class(
                total_ram_gb=16.0, cpu_count_logical=2
            )
            == "medium"
        )
        assert (
            classify_hardware_class(
                total_ram_gb=4.0, cpu_count_logical=8
            )
            == "medium"
        )

    def test_low_class(self):
        # Below the single-mode bar on either axis.
        assert (
            classify_hardware_class(
                total_ram_gb=3.5, cpu_count_logical=2
            )
            == "low"
        )
        assert (
            classify_hardware_class(
                total_ram_gb=4.0, cpu_count_logical=1
            )
            == "low"
        )

    def test_below_low_is_still_low(self):
        assert (
            classify_hardware_class(
                total_ram_gb=2.0, cpu_count_logical=1
            )
            == "low"
        )

    def test_class_is_never_none_or_empty(self):
        for ram, cores in [(1.0, 1), (64.0, 32), (5.5, 3), (8.0, 2)]:
            assert classify_hardware_class(ram, cores)


# ---------------------------------------------------------------------------
# Write + read round trip (the bundle constituent)
# ---------------------------------------------------------------------------


class TestEnvironmentArtifact:
    def test_file_name_is_contract(self):
        assert ENVIRONMENT_FILE_NAME == "environment.json"

    def test_write_and_read_roundtrip(self, tmp_path):
        path = write_environment_info(
            tmp_path,
            app_version="0.19.1",
            os_name="Windows",
            hardware_class="medium",
        )
        assert path == tmp_path / ENVIRONMENT_FILE_NAME
        data = read_environment_info(tmp_path)
        assert data["app_version"] == "0.19.1"
        assert data["os"] == "Windows"
        assert data["hardware_class"] == "medium"

    def test_schema_is_exactly_the_documented_fields(self, tmp_path):
        write_environment_info(
            tmp_path,
            app_version="0.19.1",
            os_name="Windows",
            hardware_class="high",
        )
        raw = (tmp_path / ENVIRONMENT_FILE_NAME).read_text(
            encoding="utf-8"
        )
        data = json.loads(raw)
        assert set(data) == {"app_version", "os", "hardware_class"}

    def test_prompt_flush_discipline(self, tmp_path):
        # The write must reach disk before returning (the same
        # durability contract as every capture artifact): fsync is
        # observable via a mock-wrapped os.fsync call.
        import builtins
        from unittest.mock import patch

        real_open = builtins.open

        def tracking_open(file, mode="r", *a, **kw):
            fh = real_open(file, mode, *a, **kw)
            return fh

        seen = []
        real_fsync = __import__("os").fsync

        def tracking_fsync(fd):
            seen.append(fd)
            return real_fsync(fd)

        with patch("meetandread.environment_info.os.fsync", tracking_fsync):
            write_environment_info(
                tmp_path,
                app_version="0.19.1",
                os_name="Windows",
                hardware_class="low",
            )
        assert seen, "environment.json write was not fsynced"

    def test_read_returns_none_when_absent(self, tmp_path):
        assert read_environment_info(tmp_path) is None

    def test_read_returns_none_on_corrupt_file(self, tmp_path):
        (tmp_path / ENVIRONMENT_FILE_NAME).write_text(
            "{not json", encoding="utf-8"
        )
        assert read_environment_info(tmp_path) is None

    def test_read_returns_none_on_wrong_shape(self, tmp_path):
        (tmp_path / ENVIRONMENT_FILE_NAME).write_text(
            "[1, 2, 3]", encoding="utf-8"
        )
        assert read_environment_info(tmp_path) is None

    def test_write_refuses_second_write(self, tmp_path):
        # One capture directory holds one run — a second environment
        # artifact would be a second run's (or a corruptor's). The
        # write is exclusive, like every capture artifact.
        write_environment_info(
            tmp_path,
            app_version="0.19.1",
            os_name="Windows",
            hardware_class="low",
        )
        with pytest.raises(FileExistsError):
            write_environment_info(
                tmp_path,
                app_version="0.19.2",
                os_name="Windows",
                hardware_class="low",
            )


# ---------------------------------------------------------------------------
# collect_environment_info — the reporter/wizard-side gatherer
# (stdlib-only; hardware class derivation from raw figures)
# ---------------------------------------------------------------------------


class TestCollectEnvironmentInfo:
    def test_collect_includes_stdlib_facts(self, tmp_path):
        from meetandread.environment_info import collect_environment_info

        data = collect_environment_info(
            specs_provider=lambda: (16.0, 8)
        )
        assert data["app_version"] == __import__(
            "meetandread"
        ).__version__
        assert data["os"]  # non-empty on every supported platform
        assert data["hardware_class"] == "high"

    def test_collect_survives_specs_failure(self, tmp_path):
        from meetandread.environment_info import collect_environment_info

        def boom():
            raise RuntimeError("detector broken")

        data = collect_environment_info(specs_provider=boom)
        assert data["hardware_class"] == "unknown"
