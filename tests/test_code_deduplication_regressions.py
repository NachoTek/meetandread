"""Source-level regression checks for S04 code deduplication.

These tests inspect production source files to ensure that duplicate helpers
removed in T01–T03 have not been reintroduced.  They are intentionally
narrow: they check for specific patterns that were consolidated, not general
code quality.

Gitignored paths (.gsd/, .planning/, .audits/) are never read.
"""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "meetandread"


def _read(rel: str) -> str:
    """Read a source file under ``src/meetandread/``."""
    return (SRC / rel).read_text(encoding="utf-8")


def _source_files(*globs: str) -> list[Path]:
    """Collect Python source files matching globs under ``SRC``."""
    out: list[Path] = []
    for g in globs:
        out.extend(sorted(SRC.glob(g)))
    return out


def _count_funcdef(source: str, name: str) -> int:
    """Count ``def <name>(`` occurrences (top-level or nested)."""
    return len(re.findall(rf"^\s*def\s+{re.escape(name)}\s*\(", source, re.MULTILINE))


def _has_import_statement(source: str, module: str) -> bool:
    """Return True if *source* contains an actual import of *module*."""
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if re.match(rf"^(?:from\s+)?\S*\s*import\s+.*\b{re.escape(module)}\b", stripped):
            return True
        if re.match(rf"^from\s+{re.escape(module)}\b", stripped):
            return True
    return False


# ===================================================================
# T01: Metadata footer parsing centralization
# ===================================================================


class TestMetadataFooterDedup:
    """Canonical ``parse_metadata_footer`` lives in identity_management only."""

    def test_no_private_parse_metadata_footer_in_src(self) -> None:
        """No ``def _parse_metadata_footer()`` in production code."""
        hits: list[str] = []
        for py in _source_files("**/*.py"):
            source = py.read_text(encoding="utf-8")
            if _count_funcdef(source, "_parse_metadata_footer"):
                hits.append(str(py.relative_to(ROOT)))
        assert hits == [], (
            f"Private _parse_metadata_footer found in production: {hits}"
        )

    def test_canonical_parse_metadata_footer_exists(self) -> None:
        """Canonical public ``parse_metadata_footer`` exists in identity_management."""
        source = _read("speaker/identity_management.py")
        assert _count_funcdef(source, "parse_metadata_footer") >= 1, (
            "Canonical parse_metadata_footer missing from identity_management.py"
        )


# ===================================================================
# T02: Cosine similarity centralization
# ===================================================================


class TestCosineSimilarityDedup:
    """``cosine_similarity`` is defined once in speaker/utils.py."""

    def test_no_private_cosine_similarity_in_signatures_or_diarizer(self) -> None:
        """No local ``_cosine_similarity`` in signatures.py or diarizer.py."""
        for rel in ("speaker/signatures.py", "speaker/diarizer.py"):
            source = _read(rel)
            count = _count_funcdef(source, "_cosine_similarity")
            assert count == 0, (
                f"_cosine_similarity found in {rel} (count={count})"
            )

    def test_canonical_cosine_similarity_in_speaker_utils(self) -> None:
        """Public ``cosine_similarity`` exists in speaker/utils.py."""
        source = _read("speaker/utils.py")
        assert _count_funcdef(source, "cosine_similarity") >= 1


# ===================================================================
# T02: Audio loading centralization
# ===================================================================


class TestAudioLoadingDedup:
    """``_load_audio_file`` methods must be thin wrappers delegating to canonical."""

    def test_post_processor_load_delegates(self) -> None:
        """PostProcessingQueue._load_audio_file delegates to load_wav_as_float32_mono."""
        source = _read("transcription/post_processor.py")
        # Find the method body via AST to avoid regex fragility
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_load_audio_file":
                body_src = ast.get_source_segment(source, node)
                assert body_src is not None
                assert "load_wav_as_float32_mono" in body_src, (
                    "_load_audio_file in post_processor.py does not delegate "
                    "to load_wav_as_float32_mono"
                )
                break
        else:
            pytest.fail("_load_audio_file not found in post_processor.py")

    def test_scrub_load_delegates(self) -> None:
        """ScrubRunner._load_audio_file delegates to load_wav_as_float32_mono."""
        source = _read("transcription/scrub.py")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_load_audio_file":
                body_src = ast.get_source_segment(source, node)
                assert body_src is not None
                assert "load_wav_as_float32_mono" in body_src, (
                    "_load_audio_file in scrub.py does not delegate "
                    "to load_wav_as_float32_mono"
                )
                break
        else:
            pytest.fail("_load_audio_file not found in scrub.py")

    def test_canonical_load_wav_exists(self) -> None:
        """Canonical ``load_wav_as_float32_mono`` in audio/utils.py."""
        source = _read("audio/utils.py")
        assert _count_funcdef(source, "load_wav_as_float32_mono") >= 1


# ===================================================================
# T03: _norm_label consolidation
# ===================================================================


class TestNormLabelDedup:
    """Exactly one ``_norm_label`` definition in floating_panels.py."""

    def test_single_norm_label_definition(self) -> None:
        source = _read("widgets/floating_panels.py")
        count = _count_funcdef(source, "_norm_label")
        assert count == 1, (
            f"Expected exactly 1 _norm_label in floating_panels.py, found {count}"
        )

    def test_no_norm_label_in_other_production_files(self) -> None:
        """No ``_norm_label`` outside floating_panels.py in production code."""
        hits: list[str] = []
        for py in _source_files("**/*.py"):
            if py.name == "floating_panels.py":
                continue
            source = py.read_text(encoding="utf-8")
            if _count_funcdef(source, "_norm_label"):
                hits.append(str(py.relative_to(ROOT)))
        assert hits == [], (
            f"_norm_label found outside floating_panels.py: {hits}"
        )


# ===================================================================
# T03: Dead code removal
# ===================================================================


class TestDeadCodeRemoval:
    """Dead ModelSettings.realtime_model_size field and streaming_pipeline."""

    def test_no_realtime_model_size_on_model_settings(self) -> None:
        """``realtime_model_size`` is not a direct field on ``ModelSettings``."""
        source = _read("config/models.py")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "ModelSettings":
                for item in node.body:
                    if isinstance(item, (ast.AnnAssign, ast.Assign)):
                        # Check for attribute name realtime_model_size
                        target = (
                            item.target if isinstance(item, ast.AnnAssign) else item.targets[0]
                        )
                        if isinstance(target, ast.Name) and target.id == "realtime_model_size":
                            pytest.fail(
                                "ModelSettings.realtime_model_size field still exists — "
                                "it should only be on TranscriptionSettings"
                            )
                break

    def test_streaming_pipeline_file_removed(self) -> None:
        """``streaming_pipeline.py`` does not exist."""
        assert not (SRC / "transcription" / "streaming_pipeline.py").exists(), (
            "src/meetandread/transcription/streaming_pipeline.py still exists"
        )

    def test_no_streaming_pipeline_imports(self) -> None:
        """No production or test file imports ``streaming_pipeline``."""
        hits: list[str] = []
        for pattern in ("src/**/*.py", "tests/**/*.py"):
            for py in sorted(ROOT.glob(pattern)):
                source = py.read_text(encoding="utf-8")
                if _has_import_statement(source, "streaming_pipeline"):
                    hits.append(str(py.relative_to(ROOT)))
        assert hits == [], (
            f"Import of streaming_pipeline found in: {hits}"
        )
