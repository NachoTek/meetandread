"""Tier-2 feature-dependency registry (issue #61).

A two-tier startup dependency check: Tier 1 (critical DLLs) stays fatal;
Tier 2 deps power optional features — the app runs degraded without them.
The registry is the single guidance vocabulary: the resolution text shown
in Diagnostics is the same text surfaced in dependency-stage Failed-row
details.
"""

import sys

import pytest

from meetandread import dependencies as deps
from meetandread.dependencies import (
    SHERPA_ONNX,
    DependencyStatus,
    FeatureDependency,
    check_feature_dependencies,
    dependency_error,
    dependency_failure_message,
    find_dependency,
    is_dependency_available,
    reset_availability_cache,
    unresolved_dependencies,
)


@pytest.fixture(autouse=True)
def _fresh_cache():
    reset_availability_cache()
    yield
    reset_availability_cache()


class TestRegistry:
    def test_registry_contains_sherpa_onnx(self):
        names = [dep.name for dep in deps.FEATURE_DEPENDENCIES]
        assert "sherpa-onnx" in names

    def test_sherpa_onnx_powers_speaker_identification(self):
        assert SHERPA_ONNX.module == "sherpa_onnx"
        assert SHERPA_ONNX.feature == "Speaker identification"

    def test_entries_are_immutable(self):
        with pytest.raises(Exception):
            setattr(SHERPA_ONNX, "name", "other")

    def test_find_dependency_by_name_and_module(self):
        assert find_dependency("sherpa-onnx") is SHERPA_ONNX
        assert find_dependency("sherpa_onnx") is SHERPA_ONNX
        assert find_dependency("nope") is None


class TestResolutionText:
    def test_dev_variant_mentions_install(self, monkeypatch):
        monkeypatch.delattr(sys, "frozen", raising=False)
        assert "sherpa-onnx" in SHERPA_ONNX.resolution_text()
        assert "install" in SHERPA_ONNX.resolution_text().lower()

    def test_frozen_variant_mentions_reinstall(self, monkeypatch):
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        text = SHERPA_ONNX.resolution_text()
        assert "reinstall" in text.lower()

    def test_failure_message_names_dependency_and_carries_resolution(self):
        message = dependency_failure_message(SHERPA_ONNX)
        assert SHERPA_ONNX.name in message
        assert SHERPA_ONNX.feature in message
        assert SHERPA_ONNX.resolution_text() in message


class TestAvailability:
    def test_available_when_import_succeeds(self, monkeypatch):
        monkeypatch.setattr(deps, "import_module", lambda name: object())
        assert is_dependency_available(SHERPA_ONNX) is True

    def test_unavailable_when_import_raises(self, monkeypatch):
        def boom(name):
            raise ImportError(f"No module named {name!r}")

        monkeypatch.setattr(deps, "import_module", boom)
        assert is_dependency_available(SHERPA_ONNX) is False

    def test_unavailable_when_dll_load_fails(self, monkeypatch):
        """A broken native install (DLL load failure) is unavailable too."""

        def boom(name):
            raise ImportError("DLL load failed while importing sherpa_onnx")

        monkeypatch.setattr(deps, "import_module", boom)
        assert is_dependency_available(SHERPA_ONNX) is False

    def test_import_result_is_cached_per_process(self, monkeypatch):
        calls = []

        def fake_import(name):
            calls.append(name)
            return object()

        monkeypatch.setattr(deps, "import_module", fake_import)
        assert is_dependency_available(SHERPA_ONNX) is True
        assert is_dependency_available(SHERPA_ONNX) is True
        assert calls == [SHERPA_ONNX.module]

    def test_reset_clears_the_cache(self, monkeypatch):
        monkeypatch.setattr(deps, "import_module", lambda name: object())
        assert is_dependency_available(SHERPA_ONNX) is True
        reset_availability_cache()

        def boom(name):
            raise ImportError(name)

        monkeypatch.setattr(deps, "import_module", boom)
        assert is_dependency_available(SHERPA_ONNX) is False


class TestCheckFeatureDependencies:
    def test_reports_status_for_every_registered_dep(self, monkeypatch):
        monkeypatch.setattr(deps, "import_module", lambda name: object())
        statuses = check_feature_dependencies()
        assert [s.dependency.name for s in statuses] == [
            dep.name for dep in deps.FEATURE_DEPENDENCIES
        ]
        assert all(s.available for s in statuses)

    def test_unresolved_lists_only_missing(self, monkeypatch):
        extra = FeatureDependency(
            name="always-missing",
            module="always_missing_module",
            feature="Nothing",
            resolution_dev="dev",
            resolution_frozen="frozen",
        )
        monkeypatch.setattr(
            deps, "FEATURE_DEPENDENCIES", (SHERPA_ONNX, extra)
        )

        def fake_import(name):
            if name == extra.module:
                raise ImportError(name)
            return object()

        monkeypatch.setattr(deps, "import_module", fake_import)

        unresolved = unresolved_dependencies()
        assert [s.dependency.name for s in unresolved] == ["always-missing"]


class TestDependencyError:
    def test_error_carries_failure_message_and_dependency_name(self):
        error = dependency_error(SHERPA_ONNX)
        assert isinstance(error, ImportError)
        assert str(error) == dependency_failure_message(SHERPA_ONNX)
        assert getattr(error, "dependency_name") == SHERPA_ONNX.name

    def test_status_dataclass_round_trip(self):
        status = DependencyStatus(dependency=SHERPA_ONNX, available=False)
        assert status.dependency is SHERPA_ONNX
        assert status.available is False
