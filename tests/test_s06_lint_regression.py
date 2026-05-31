"""S06 lint-regression guard: final cleanup contract.

Uses Python AST and source inspection over ``src/meetandread`` only.
Guards that:
  1. No standalone dead ``pass`` remains in the controller cleanup block.
  2. No stale ``# noqa: F841``, ``# noqa: F541``, or ``# noqa: F811``
     comments remain in production source beyond the documented allowlist.
  3. No executable ``print()`` calls or ``assert`` statements exist
     (mirrors S05 but scoped to S06-touched files for fast feedback).
  4. No non-interpolating f-strings carry an F541 suppression that could
     simply be converted to regular strings.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "meetandread"

# Files directly touched by S06 tasks T01–T02.
S06_TOUCHED_FILES: List[str] = [
    "recording/controller.py",
    "audio/capture/devices.py",
    "main.py",
    "widgets/floating_panels.py",
]

# ---------------------------------------------------------------------------
# Allowlists — noqa suppressions that are intentionally kept
# ---------------------------------------------------------------------------

# {relative_path: {line_number: reason}}
NOQA_ALLOWLIST: Dict[str, Dict[int, str]] = {
    "widgets/floating_panels.py": {
        # Intentional f-string using only escaped braces for Qt CSS template.
        5323: "F541 — CSS template with only escaped braces, kept for clarity",
        # Method call incorrectly flagged by stale noqa; kept as documentation
        # until a future cleanup removes the comment itself.
        7554: "F841 — stale suppression on method call (harmless comment)",
        # Intentional redefinition in ``if __name__ == "__main__":`` block.
        9214: "F811 — intentional redefinition in standalone test block",
    },
}


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


def _find_noqa_comments(filepath: Path) -> List[Tuple[int, str, str]]:
    """Return (lineno, noqa_code, full_line) for F841/F541/F811 suppressions."""
    hits: List[Tuple[int, str, str]] = []
    source = filepath.read_text(encoding="utf-8")
    for i, line in enumerate(source.splitlines(), 1):
        for code in ("F841", "F541", "F811"):
            if re.search(rf"#\s*noqa:\s*{code}\b", line):
                hits.append((i, code, line.strip()))
    return hits


def _find_dead_pass(tree: ast.Module) -> List[Tuple[int, str]]:
    """Return (lineno, parent_type) for blocks whose only body is ``pass``.

    Excludes ``except`` handlers (which sometimes use pass legitimately)
    and abstract base class placeholders.
    """
    hits: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            # except-handler pass is acceptable for intentional swallowing
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                             ast.For, ast.While, ast.If, ast.With, ast.Try)):
            if (
                len(node.body) == 1
                and isinstance(node.body[0], ast.Pass)
            ):
                hits.append((node.lineno, node.__class__.__name__))
    return hits


def _find_print_calls(tree: ast.Module) -> List[Tuple[int, str]]:
    """Return (lineno, func_name) for every executable print() call."""
    hits: List[Tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "print":
                hits.append((node.lineno, "print"))
            elif (
                isinstance(func, ast.Attribute)
                and func.attr == "print"
                and isinstance(func.value, ast.Name)
                and func.value.id == "builtins"
            ):
                hits.append((node.lineno, "builtins.print"))
    return hits


def _find_asserts(tree: ast.Module) -> List[int]:
    """Return line numbers of all ast.Assert nodes."""
    return [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assert)]


# ---------------------------------------------------------------------------
# Test 1: No dead pass in controller cleanup block
# ---------------------------------------------------------------------------


class TestNoDeadPass:
    """Guard: no standalone dead ``pass`` in the controller cleanup block."""

    def test_controller_no_dead_pass(self):
        fp = SRC_ROOT / "recording" / "controller.py"
        assert fp.exists(), f"Expected file not found: {fp}"
        tree = _parse_file(fp)
        hits = _find_dead_pass(tree)
        assert not hits, (
            "Found dead ``pass`` statements in controller.py "
            "(blocks whose only body is pass):\n"
            + "\n".join(f"  line {ln}: {parent}" for ln, parent in hits)
        )

    def test_no_dead_pass_in_s06_touched_files(self):
        """Extend dead-pass check to all S06-touched files."""
        violations: List[str] = []
        for rel in S06_TOUCHED_FILES:
            fp = SRC_ROOT / rel
            if not fp.exists():
                continue
            tree = _parse_file(fp)
            for ln, parent in _find_dead_pass(tree):
                violations.append(f"{rel}:{ln}: {parent}")
        assert not violations, (
            "Found dead ``pass`` in S06-touched files:\n"
            + "\n".join(f"  {v}" for v in violations)
        )


# ---------------------------------------------------------------------------
# Test 2: No stale noqa F841/F541/F811 beyond allowlist
# ---------------------------------------------------------------------------


class TestNoStaleNoqaSuppressions:
    """Guard: no stale noqa suppressions in production source.

    Known-justified suppressions are allowlisted above.  Any new
    suppressions must either be removed (preferred) or added to the
    allowlist with a documented reason.
    """

    def test_noqa_suppressions_match_allowlist(self):
        violations: List[str] = []
        for fp in _collect_py_files(SRC_ROOT):
            rel = str(fp.relative_to(SRC_ROOT)).replace(os.sep, "/")
            allowed = NOQA_ALLOWLIST.get(rel, {})
            for lineno, code, text in _find_noqa_comments(fp):
                if lineno in allowed:
                    continue
                violations.append(f"{rel}:{lineno}: {code} — {text}")
        assert not violations, (
            "Found noqa F841/F541/F811 suppressions not in allowlist.\n"
            "Either remove the suppression or add to NOQA_ALLOWLIST "
            "with a reason:\n"
            + "\n".join(f"  {v}" for v in violations)
        )


# ---------------------------------------------------------------------------
# Test 3: No print/assert in S06-touched files (fast feedback)
# ---------------------------------------------------------------------------


class TestS06TouchedFileSanity:
    """Focused guard: no print/assert in S06-touched files.

    S05 already covers the full source tree; this is a fast
    subset check scoped to S06-touched files for quick feedback.
    """

    def test_no_print_in_s06_files(self):
        violations: List[str] = []
        for rel in S06_TOUCHED_FILES:
            fp = SRC_ROOT / rel
            if not fp.exists():
                continue
            tree = _parse_file(fp)
            for ln, name in _find_print_calls(tree):
                violations.append(f"{rel}:{ln}: {name}")
        assert not violations, (
            "Found print() calls in S06-touched files:\n"
            + "\n".join(f"  {v}" for v in violations)
        )

    def test_no_assert_in_s06_files(self):
        violations: List[str] = []
        for rel in S06_TOUCHED_FILES:
            fp = SRC_ROOT / rel
            if not fp.exists():
                continue
            tree = _parse_file(fp)
            for ln in _find_asserts(tree):
                violations.append(f"{rel}:{ln}")
        assert not violations, (
            "Found assert statements in S06-touched files:\n"
            + "\n".join(f"  {v}" for v in violations)
        )


# ---------------------------------------------------------------------------
# Test 4: No non-interpolating f-string with F541 that could be a regular string
# ---------------------------------------------------------------------------


class TestNoUnnecessaryF541:
    """Guard: F541-suppressed f-strings should actually need to be f-strings.

    If a line carries ``# noqa: F541`` but the f-string has no
    ``FormattedValue`` nodes (no ``{expr}`` interpolation), it could be
    a regular string — and the noqa is covering a genuine code smell.
    """

    def test_f541_suppressions_are_genuine_fstrings(self):
        violations: List[str] = []
        for fp in _collect_py_files(SRC_ROOT):
            rel = str(fp.relative_to(SRC_ROOT)).replace(os.sep, "/")
            allowed = NOQA_ALLOWLIST.get(rel, {})
            source = fp.read_text(encoding="utf-8")
            lines = source.splitlines()
            try:
                tree = ast.parse(source, filename=str(fp))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.JoinedStr):
                    continue
                lineno = node.lineno
                # Only check lines that carry an F541 suppression
                if lineno > len(lines):
                    continue
                line_text = lines[lineno - 1]
                if not re.search(r"#\s*noqa:\s*F541\b", line_text):
                    continue
                # Skip allowlisted entries
                if lineno in allowed:
                    continue
                has_interpolation = any(
                    isinstance(v, ast.FormattedValue) for v in node.values
                )
                if not has_interpolation:
                    violations.append(
                        f"{rel}:{lineno}: F541 on non-interpolating f-string "
                        f"— convert to regular string"
                    )
        assert not violations, (
            "Found F541 suppressions on non-interpolating f-strings:\n"
            + "\n".join(f"  {v}" for v in violations)
        )
