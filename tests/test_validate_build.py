"""Tests for validate_build.py data-file checks (issue #71).

PyInstaller 6.x places bundled datas under ``dist/<name>/_internal/...``
while older layouts placed them directly under ``dist/<name>/...``. The
SVG-icon and performance-test-data checks must tolerate both layouts,
the same way the DLL checks already do (recursive search from the
bundle root).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def validate_build() -> ModuleType:
    """Load the repo-root validate_build.py as a fresh module."""
    spec = importlib.util.spec_from_file_location(
        "validate_build", ROOT / "validate_build.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_bundle(
    tmp_path: Path,
    *,
    svg: str | None = None,
    test_data: str | None = None,
) -> Path:
    """Create ``dist/meetandread`` populated with optional datas.

    ``svg``/``test_data`` are bundle-relative file paths to create, e.g.
    ``"_internal/meetandread/widgets/icon.svg"``.
    """
    bundle = tmp_path / "dist" / "meetandread"
    for relative in (svg, test_data):
        if relative is not None:
            target = bundle / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"data")
    return bundle


def _internal_svg() -> str:
    return "_internal/meetandread/widgets/recording.svg"


def _internal_test_data() -> str:
    return "_internal/meetandread/performance/test_data/sample.wav"


class TestCheckTestData:
    """check_test_data must find datas in any PyInstaller layout."""

    def test_finds_datas_in_internal_layout(
        self, validate_build, tmp_path
    ):
        """PyInstaller 6.x relocates datas under _internal/ — must pass."""
        bundle = _make_bundle(
            tmp_path, svg=_internal_svg(), test_data=_internal_test_data()
        )
        validate_build.BUILD_DIR = str(bundle)
        assert validate_build.check_test_data() is True

    def test_finds_datas_in_legacy_root_layout(
        self, validate_build, tmp_path
    ):
        """Pre-6.x bundles place datas directly under the bundle root."""
        bundle = _make_bundle(
            tmp_path,
            svg="meetandread/widgets/recording.svg",
            test_data="meetandread/performance/test_data/sample.wav",
        )
        validate_build.BUILD_DIR = str(bundle)
        assert validate_build.check_test_data() is True

    def test_missing_svg_icons_fails(self, validate_build, tmp_path):
        bundle = _make_bundle(tmp_path, test_data=_internal_test_data())
        validate_build.BUILD_DIR = str(bundle)
        assert validate_build.check_test_data() is False

    def test_missing_test_data_fails(self, validate_build, tmp_path):
        bundle = _make_bundle(tmp_path, svg=_internal_svg())
        validate_build.BUILD_DIR = str(bundle)
        assert validate_build.check_test_data() is False

    def test_empty_bundle_fails(self, validate_build, tmp_path):
        bundle = _make_bundle(tmp_path)
        validate_build.BUILD_DIR = str(bundle)
        assert validate_build.check_test_data() is False
