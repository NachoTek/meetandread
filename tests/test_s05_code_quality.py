"""S05 code-quality guard: executable contract for no-print, no-assert, no-footer-find.

Uses Python AST and source inspection over src/meetandread only.
Guards that:
  1. No executable ast.Call to builtin print() in production source.
  2. No ast.Assert nodes in production source.
  3. No usage of content.find(_FOOTER_MARKER) — must use rfind per T01 fix.
  4. No silent ``except Exception: pass`` in S05-touched files.
"""

from __future__ import annotations

import ast
import os
import re
import textwrap
from pathlib import Path
from typing import List, Tuple

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "meetandread"

# Files directly touched by S05 tasks T01–T04.
S05_TOUCHED_FILES: List[str] = [
    "speaker/identity_management.py",
    "recording/controller.py",
    "speaker/signatures.py",
    "speaker/diarizer.py",
    "audio/session.py",
    "audio/capture/devices.py",
    "audio/cli.py",
    "audio/storage/paths.py",
    "audio/storage/recovery.py",
    "audio/storage/wav_finalize.py",
    "hardware/detector.py",
    "hardware/recommender.py",
    "main.py",
    "performance/benchmark.py",
    "performance/monitor.py",
    "transcription/accumulating_processor.py",
    "transcription/engine.py",
    "transcription/local_agreement.py",
    "transcription/post_processor.py",
]


def _collect_py_files(root: Path) -> List[Path]:
    """Collect all .py files under *root*, excluding __pycache__."""
    paths: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "__pycache__" in dirpath:
            continue
        for fn in filenames:
            if fn.endswith(".py"):
                paths.append(Path(dirpath) / fn)
    return sorted(paths)


def _parse_file(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _find_print_calls(tree: ast.Module, filepath: Path) -> List[Tuple[int, str]]:
    """Return (lineno, text) for every executable print() call."""
    hits: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            is_print = False
            if isinstance(func, ast.Name) and func.id == "print":
                is_print = True
            elif (
                isinstance(func, ast.Attribute)
                and func.attr == "print"
                and isinstance(func.value, ast.Name)
                and func.value.id == "builtins"
            ):
                is_print = True
            if is_print:
                hits.append((node.lineno, ast.get_source_segment(open(filepath, encoding="utf-8").read(), node) or "<print call>"))
    return hits


def _find_asserts(tree: ast.Module) -> List[int]:
    """Return line numbers of all ast.Assert nodes."""
    return [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assert)]


def _find_footer_find_usage(filepath: Path) -> List[Tuple[int, str]]:
    """Find lines using .find( with _FOOTER_MARKER (the pre-T01 bug pattern).

    The correct pattern is .rfind(_FOOTER_MARKER).  A plain .find(
    would locate the *first* occurrence in the file body, which could be
    a false marker embedded in transcript content.
    """
    hits: List[Tuple[int, str]] = []
    source = filepath.read_text(encoding="utf-8")
    for i, line in enumerate(source.splitlines(), 1):
        # Match .find( preceded by any text and followed by _FOOTER_MARKER
        # but NOT .rfind(
        if "_FOOTER_MARKER" in line and ".find(" in line and ".rfind(" not in line:
            hits.append((i, line.strip()))
    return hits


def _find_silent_broad_except(tree: ast.Module) -> List[Tuple[int, str]]:
    """Return (lineno, except_clause_text) for bare ``except Exception: pass``."""
    hits: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            if node.type and isinstance(node.type, ast.Name) and node.type.id == "Exception":
                # Body is only a Pass statement
                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    hits.append((node.lineno, "except Exception: pass"))
    return hits


# ---------------------------------------------------------------------------
# Test 1: No executable print() in production source
# ---------------------------------------------------------------------------


class TestNoExecutablePrint:
    """AST-level guard: no executable print() calls in src/meetandread."""

    @pytest.fixture(autouse=True)
    def _collect(self):
        self.files = _collect_py_files(SRC_ROOT)

    def test_no_print_calls_in_production(self):
        violations: List[str] = []
        for fp in self.files:
            tree = _parse_file(fp)
            for lineno, text in _find_print_calls(tree, fp):
                violations.append(f"{fp.relative_to(SRC_ROOT.parent.parent)}:{lineno}: {text}")

        assert not violations, (
            "Found executable print() calls in production source:\n"
            + "\n".join(f"  {v}" for v in violations)
        )


# ---------------------------------------------------------------------------
# Test 2: No assert statements in production source
# ---------------------------------------------------------------------------


class TestNoRuntimeAssert:
    """AST-level guard: no assert statements in src/meetandread."""

    @pytest.fixture(autouse=True)
    def _collect(self):
        self.files = _collect_py_files(SRC_ROOT)

    def test_no_assert_in_production(self):
        violations: List[str] = []
        for fp in self.files:
            tree = _parse_file(fp)
            for lineno in _find_asserts(tree):
                violations.append(f"{fp.relative_to(SRC_ROOT.parent.parent)}:{lineno}")

        assert not violations, (
            "Found assert statements in production source:\n"
            + "\n".join(f"  {v}" for v in violations)
        )


# ---------------------------------------------------------------------------
# Test 3: No content.find(_FOOTER_MARKER) — must use rfind
# ---------------------------------------------------------------------------


class TestFooterFindGuard:
    """Source guard: no usage of .find(_FOOTER_MARKER) (must use .rfind)."""

    def test_no_footer_find_in_identity_management(self):
        fp = SRC_ROOT / "speaker" / "identity_management.py"
        assert fp.exists(), f"Expected file not found: {fp}"
        hits = _find_footer_find_usage(fp)
        assert not hits, (
            "Found .find(_FOOTER_MARKER) — use .rfind(_FOOTER_MARKER) instead "
            "to match canonical parse_metadata_footer behavior:\n"
            + "\n".join(f"  line {ln}: {text}" for ln, text in hits)
        )

    def test_no_footer_find_anywhere_in_src(self):
        """Check all production source for the footer-find bug pattern."""
        violations: List[str] = []
        for fp in _collect_py_files(SRC_ROOT):
            for lineno, text in _find_footer_find_usage(fp):
                violations.append(f"{fp.relative_to(SRC_ROOT.parent.parent)}:{lineno}: {text}")

        assert not violations, (
            "Found .find(_FOOTER_MARKER) — use .rfind(_FOOTER_MARKER) instead:\n"
            + "\n".join(f"  {v}" for v in violations)
        )


# ---------------------------------------------------------------------------
# Test 4: No silent ``except Exception: pass`` in S05-touched files
# ---------------------------------------------------------------------------


class TestNoSilentBroadExcept:
    """Guard: no bare ``except Exception: pass`` in files touched by S05.

    Crash guards that intentionally swallow exceptions must at minimum log
    at debug level — they should never be a bare ``pass``.
    """

    @pytest.fixture(autouse=True)
    def _collect(self):
        self.files = []
        for rel in S05_TOUCHED_FILES:
            fp = SRC_ROOT / rel
            if fp.exists():
                self.files.append(fp)

    def test_no_silent_broad_except_in_s05_files(self):
        violations: List[str] = []
        for fp in self.files:
            tree = _parse_file(fp)
            for lineno, text in _find_silent_broad_except(tree):
                violations.append(f"{fp.relative_to(SRC_ROOT.parent.parent)}:{lineno}: {text}")

        assert not violations, (
            "Found silent ``except Exception: pass`` in S05-touched files "
            "(must log at minimum debug level):\n"
            + "\n".join(f"  {v}" for v in violations)
        )
