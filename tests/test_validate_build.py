"""Tests for the PyInstaller build validator (issue #71).

``validate_build.py`` historically globbed data files at fixed paths under
``dist/meetandread/meetandread/...``. PyInstaller 6.x relocates datas to
``dist/meetandread/_internal/...``, which made the SVG and performance
test-data checks fail on every CI run since 2026-07-31. These tests pin the
new contract: data-file checks must locate files by recursive search from
the bundle root (like the DLL checks), tolerant of ``_internal/``
relocation.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
VALIDATE_BUILD = ROOT / "validate_build.py"


def _load_validate_build():
    """Import validate_build.py by path (it lives at the repo root)."""
    spec = importlib.util.spec_from_file_location("validate_build_under_test", VALIDATE_BUILD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def validator():
    return _load_validate_build()


@pytest.fixture()
def bundle(tmp_path: Path) -> Path:
    """A minimal PyInstaller-6-style bundle: datas under ``_internal/``."""
    build_dir = tmp_path / "dist" / "meetandread"
    (build_dir / "_internal" / "meetandread" / "widgets").mkdir(parents=True)
    (build_dir / "_internal" / "meetandread" / "performance" / "test_data").mkdir(parents=True)
    (build_dir / "meetandread.exe").write_bytes(b"fake exe")
    (build_dir / "_internal" / "meetandread" / "widgets" / "mic.svg").write_text("<svg/>")
    (build_dir / "_internal" / "meetandread" / "performance" / "test_data" / "results.txt").write_text("data")
    return build_dir


class TestCheckTestData:
    """Data-file checks must be layout-tolerant (issue #71)."""

    def test_finds_assets_under_internal_layout(self, validator, bundle, capsys):
        """PyInstaller 6.x places datas under ``_internal/`` — must pass."""
        assert validator.check_test_data(str(bundle)) is True
        out = capsys.readouterr().out
        assert "SVG" in out

    def test_finds_assets_under_legacy_layout(self, validator, bundle, tmp_path):
        """Pre-6.x layout (datas directly under the bundle root) must also pass."""
        legacy = tmp_path / "dist" / "legacy"
        (legacy / "meetandread" / "widgets").mkdir(parents=True)
        (legacy / "meetandread" / "performance" / "test_data").mkdir(parents=True)
        (legacy / "meetandread" / "widgets" / "mic.svg").write_text("<svg/>")
        (legacy / "meetandread" / "performance" / "test_data" / "results.txt").write_text("data")

        assert validator.check_test_data(str(legacy)) is True

    def test_fails_when_svgs_missing(self, validator, bundle, capsys):
        (bundle / "_internal" / "meetandread" / "widgets" / "mic.svg").unlink()
        assert validator.check_test_data(str(bundle)) is False
        assert "SVG" in capsys.readouterr().out

    def test_fails_when_test_data_missing(self, validator, bundle, capsys):
        (bundle / "_internal" / "meetandread" / "performance" / "test_data" / "results.txt").unlink()
        assert validator.check_test_data(str(bundle)) is False


class TestCheckBuildDirectory:
    def test_missing_directory_fails(self, validator, tmp_path, capsys):
        assert validator.check_build_directory(str(tmp_path / "nope")) is False

    def test_existing_directory_passes(self, validator, bundle):
        assert validator.check_build_directory(str(bundle)) is True


class TestCheckRequiredDlls:
    def test_recursive_search_finds_dll_under_internal(self, validator, bundle):
        """DLL patterns are searched recursively from the bundle root."""
        (bundle / "_internal" / "portaudio.dll").write_bytes(b"dll")
        assert validator.check_required_dlls(str(bundle)) is False  # other patterns still missing

        # Provide every remaining required pattern; must now pass.
        for pattern in validator.REQUIRED_LIBRARIES:
            rel = pattern if "/" in pattern else f"{pattern}x.dll"
            target = bundle / "_internal" / rel
            if not target.exists():
                if "/" in pattern:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(b"dll")
                else:
                    target.write_bytes(b"dll")
        assert validator.check_required_dlls(str(bundle)) is True
