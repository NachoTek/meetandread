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
# Issue #50: Transcript Footer parsing centralized in transcript_footer
# ===================================================================


class TestMetadataFooterDedup:
    """Transcript Footer parsing and marker literals live only in the
    canonical ``transcript_footer`` module.

    Issue #50 deleted the duplicate footer parsers, splitters, rebuilders,
    marker constants, and the four-value parsing contract from the Speaker
    identity modules.  These guards keep them from returning.
    """

    def test_no_footer_parsers_in_speaker_modules(self) -> None:
        """No footer parser/splitter/rebuilder defined anywhere under speaker/."""
        forbidden = (
            "parse_metadata_footer",
            "split_metadata_footer",
            "_parse_metadata_footer",
            "_rebuild_transcript",
            "_rebuild_file",
        )
        hits: list[str] = []
        for py in _source_files("speaker/**/*.py"):
            source = py.read_text(encoding="utf-8")
            for name in forbidden:
                if _count_funcdef(source, name):
                    hits.append(f"{py.relative_to(ROOT)} defines {name}")
        assert hits == [], (
            f"Footer parser found in speaker modules: {hits}"
        )

    def test_no_footer_marker_literals_in_production_outside_canonical(self) -> None:
        """The '<!-- METADATA' marker literal appears only in transcript_footer.py.

        This is the acceptance guard for issue #50: Transcript Footer syntax
        is owned by the canonical module alone (intentional test fixtures are
        out of scope — this scans production source only).
        """
        hits: list[str] = []
        for py in _source_files("**/*.py"):
            if py.name == "transcript_footer.py":
                continue
            source = py.read_text(encoding="utf-8")
            if "<!-- METADATA" in source or "_FOOTER_MARKER" in source:
                hits.append(str(py.relative_to(ROOT)))
        assert hits == [], (
            f"Transcript Footer syntax found outside the canonical module: {hits}"
        )

    def test_canonical_footer_module_exposes_four_operations(self) -> None:
        """transcript_footer.py defines the four public operations."""
        source = _read("transcription/transcript_footer.py")
        for op in ("parse", "split", "join", "strip"):
            assert _count_funcdef(source, op) >= 1, (
                f"Canonical transcript_footer.{op} missing"
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

    def test_retranscribe_load_delegates(self) -> None:
        """RetranscribeRunner._load_audio_file delegates to load_wav_as_float32_mono."""
        source = _read("transcription/retranscribe.py")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_load_audio_file":
                body_src = ast.get_source_segment(source, node)
                assert body_src is not None
                assert "load_wav_as_float32_mono" in body_src, (
                    "_load_audio_file in retranscribe.py does not delegate "
                    "to load_wav_as_float32_mono"
                )
                break
        else:
            pytest.fail("_load_audio_file not found in retranscribe.py")

    def test_canonical_load_wav_exists(self) -> None:
        """Canonical ``load_wav_as_float32_mono`` in audio/utils.py."""
        source = _read("audio/utils.py")
        assert _count_funcdef(source, "load_wav_as_float32_mono") >= 1


# ===================================================================
# T03: _norm_label consolidation — moved to speaker/identity_linking.py
# ===================================================================


class TestNormLabelDedup:
    """Exactly one ``_norm_label`` definition in speaker/identity_linking.py."""

    def test_single_norm_label_definition(self) -> None:
        source = _read("speaker/identity_linking.py")
        count = _count_funcdef(source, "_norm_label")
        assert count == 1, (
            f"Expected exactly 1 _norm_label in speaker/identity_linking.py, found {count}"
        )

    def test_no_norm_label_in_other_production_files(self) -> None:
        """No ``_norm_label`` outside speaker/identity_linking.py in production code."""
        hits: list[str] = []
        for py in _source_files("**/*.py"):
            if py.name == "identity_linking.py":
                continue
            source = py.read_text(encoding="utf-8")
            if _count_funcdef(source, "_norm_label"):
                hits.append(str(py.relative_to(ROOT)))
        assert hits == [], (
            f"_norm_label found outside speaker/identity_linking.py: {hits}"
        )


# ===================================================================
# Issue #44: Speaker rename wrappers eliminated from both panels
# ===================================================================


class TestRenameWrapperRemoval:
    """Neither panel class defines its own rename/propagate helper.

    Issue #40 extracted the logic into ``speaker.identity_linking``; issue #44
    removes the one-line panel wrappers that were left as migration scaffolding.
    Callers must invoke the canonical module functions directly.
    """

    @pytest.mark.parametrize("method", ["_rename_speaker_in_file", "_propagate_rename_to_signatures"])
    def test_panels_define_no_rename_wrapper(self, method: str) -> None:
        source = _read("widgets/floating_panels.py")
        count = _count_funcdef(source, method)
        assert count == 0, (
            f"{method} is still defined in floating_panels.py ({count}x) — "
            f"callers must use speaker.identity_linking directly"
        )


# ===================================================================
# Issue #45: signature-propagation helpers deduplicated
# ===================================================================


class TestSignaturePropagationDedup:
    """The link- and rename-propagators share one db-resolution + store-interaction
    helper (issue #45).  These structural guards keep the shape from re-diverging.
    """

    def test_voice_signature_store_import_appears_once(self) -> None:
        """The lazy VoiceSignatureStore import lives only in the shared helper."""
        source = _read("speaker/identity_linking.py")
        count = source.count("from meetandread.speaker.signatures import VoiceSignatureStore")
        assert count == 1, (
            f"Expected 1 VoiceSignatureStore import in identity_linking.py, found {count}"
        )

    def test_recordings_dir_resolver_appears_once(self) -> None:
        """The recordings-dir fallback lives only in _resolve_signature_db."""
        source = _read("speaker/identity_linking.py")
        count = source.count("from meetandread.audio.storage.paths import get_recordings_dir")
        assert count == 1, (
            f"Expected 1 get_recordings_dir import in identity_linking.py, found {count}"
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
