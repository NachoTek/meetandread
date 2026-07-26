"""Packaging consistency regression checks (Issue #34).

These tests guard the two artifacts that broke when the backward-compat
``scrub`` shim was removed (commit 2446285):

* ``meetandread.spec`` (the PyInstaller bundling spec) kept listing
  ``meetandread.transcription.scrub`` — a module that no longer exists —
  so a packaged build could not complete.
* ``src/meetandread.egg-info/PKG-INFO`` (the packaged metadata, derived
  from ``README.md`` and self-healing on build) still advertised the
  pre-rename terminology ("Audio Capture", "Transcript scrub", "History
  tab", "Audio Input").

They also assert that the bundling spec and the build-validation
entrypoint (``validate_build.py``) stay consistent, so the two lists of
application modules cannot drift apart again.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "meetandread.spec"
VALIDATE_BUILD = ROOT / "validate_build.py"
PKG_INFO = ROOT / "src" / "meetandread.egg-info" / "PKG-INFO"
SOURCES_TXT = ROOT / "src" / "meetandread.egg-info" / "SOURCES.txt"
SRC_ROOT = ROOT / "src" / "meetandread"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_literal(source: str, name: str) -> list[str]:
    """Return the string elements of a top-level ``name = [...]`` assignment."""
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == name and isinstance(node.value, ast.List):
                return [
                    elt.value
                    for elt in node.value.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                ]
    pytest.fail(f"No top-level list assignment named {name!r} found")


def _spec_hiddenimports() -> list[str]:
    return _list_literal(SPEC.read_text(encoding="utf-8"), "hiddenimports")


def _validate_required_imports() -> list[str]:
    return _list_literal(VALIDATE_BUILD.read_text(encoding="utf-8"), "REQUIRED_IMPORTS")


def _module_to_path(dotted: str) -> Path:
    """Resolve a ``meetandread.*`` dotted module to its on-disk source file."""
    rel = dotted.replace(".", "/")
    py_file = (SRC_ROOT / rel[len("meetandread/"):]).with_suffix(".py")
    if py_file.exists():
        return py_file
    return SRC_ROOT / rel[len("meetandread/"):] / "__init__.py"


# ---------------------------------------------------------------------------
# Bundling spec
# ---------------------------------------------------------------------------

class TestBundlingSpec:
    """The PyInstaller spec must only reference modules that still exist."""

    def test_retranscribe_is_listed_not_scrub(self):
        """The spec lists the current transcription module, not the deleted one."""
        hidden = _spec_hiddenimports()
        assert "meetandread.transcription.retranscribe" in hidden
        assert "meetandread.transcription.scrub" not in hidden

    def test_no_deleted_module_referenced(self):
        """No hidden import may mention the removed ``scrub`` shim anywhere."""
        hidden = _spec_hiddenimports()
        offenders = [m for m in hidden if "scrub" in m]
        assert offenders == [], f"spec still references deleted modules: {offenders}"

    def test_every_app_hiddenimport_resolves_to_a_file(self):
        """Every ``meetandread.*`` hidden import must map to a real source file.

        Generalises the scrub regression: any future module rename or deletion
        that forgets the spec will fail here.
        """
        hidden = _spec_hiddenimports()
        missing = [
            m for m in hidden
            if m.startswith("meetandread.") and not _module_to_path(m).exists()
        ]
        assert missing == [], f"spec references non-existent modules: {missing}"


# ---------------------------------------------------------------------------
# Spec <-> validate_build.py consistency
# ---------------------------------------------------------------------------

class TestSpecAndValidationAgree:
    """The bundling spec and the validation entrypoint must not drift apart."""

    def test_app_module_lists_match(self):
        """The ``meetandread.*`` modules listed in both files must be identical."""
        spec_modules = {
            m for m in _spec_hiddenimports() if m.startswith("meetandread")
        }
        validate_modules = {
            m for m in _validate_required_imports() if m.startswith("meetandread")
        }
        assert spec_modules == validate_modules, (
            "meetandread.spec hiddenimports and validate_build.py REQUIRED_IMPORTS "
            f"disagree on application modules.\n"
            f"  only in spec:   {sorted(spec_modules - validate_modules)}\n"
            f"  only in validate: {sorted(validate_modules - spec_modules)}"
        )


# ---------------------------------------------------------------------------
# Packaged metadata
# ---------------------------------------------------------------------------

class TestPackagedMetadata:
    """``PKG-INFO`` is derived from README.md and must not ship stale terms."""

    STALE_TERMS = ["Audio Capture", "scrub", "History tab", "Audio Input"]

    def test_pkg_info_reflects_current_readme(self):
        """PKG-INFO embeds the current README body (self-heals on build)."""
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        pkg_info = PKG_INFO.read_text(encoding="utf-8")
        assert readme in pkg_info, (
            "PKG-INFO body does not contain the current README.md — run "
            "`python -m build` (or `python setup.py egg_info`) to regenerate."
        )

    def test_pkg_info_has_no_stale_terminology(self):
        """Packaged metadata must not advertise pre-rename domain terms."""
        pkg_info = PKG_INFO.read_text(encoding="utf-8")
        present = [term for term in self.STALE_TERMS if term in pkg_info]
        assert present == [], f"PKG-INFO still advertises stale terms: {present}"

    def test_sources_txt_has_no_deleted_scrub_module(self):
        """SOURCES.txt must not advertise the deleted ``scrub`` source files."""
        sources = SOURCES_TXT.read_text(encoding="utf-8")
        assert "transcription/scrub.py" not in sources
        assert "test_scrub.py" not in sources
