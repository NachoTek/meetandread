"""Tests for the RetranscribeRunner background re-transcription module.

This is the new domain-aligned name for the runner previously called
``ScrubRunner``. The legacy class is preserved as a backward-compat shim
re-exported from ``scrub.py`` and is still covered by ``test_scrub.py``.
"""

from pathlib import Path

import pytest

from meetandread.config.models import AppSettings
from meetandread.transcription.retranscribe import RetranscribeRunner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settings() -> AppSettings:
    return AppSettings()


@pytest.fixture
def transcript_path(tmp_path: Path) -> Path:
    """A fake canonical transcript file."""
    p = tmp_path / "recording-2026-04-26-120000.md"
    p.write_text("# Transcript\n\nHello world\n")
    return p


# ---------------------------------------------------------------------------
# Module + shim exports
# ---------------------------------------------------------------------------

class TestModuleShape:
    def test_retranscribe_runner_exists(self):
        from meetandread.transcription import retranscribe
        assert hasattr(retranscribe, "RetranscribeRunner")

    def test_scrub_module_re_exports_retranscribe_runner(self):
        """scrub.py is a backward-compat shim that re-exports RetranscribeRunner."""
        from meetandread.transcription import scrub
        assert hasattr(scrub, "RetranscribeRunner")

    def test_scrub_module_still_exports_scrub_runner(self):
        """ScrubRunner remains importable from scrub.py for legacy callers."""
        from meetandread.transcription import scrub
        assert hasattr(scrub, "ScrubRunner")

    def test_scrub_runner_is_distinct_class_for_backward_compat(self):
        """ScrubRunner is a real class (subclass) that produces legacy _scrub_ names."""
        from meetandread.transcription.scrub import ScrubRunner
        # ScrubRunner must be related to RetranscribeRunner (sharing logic)
        assert issubclass(ScrubRunner, RetranscribeRunner)


# ---------------------------------------------------------------------------
# Sidecar naming — uses _retranscribe_ tag
# ---------------------------------------------------------------------------

class TestSidecarNaming:
    def test_basic_naming_uses_retranscribe_tag(self, transcript_path: Path):
        result = RetranscribeRunner.sidecar_path(transcript_path, "small")
        assert result == transcript_path.parent / (
            "recording-2026-04-26-120000_retranscribe_small.md"
        )

    def test_different_models_produce_different_paths(self, transcript_path: Path):
        s1 = RetranscribeRunner.sidecar_path(transcript_path, "tiny")
        s2 = RetranscribeRunner.sidecar_path(transcript_path, "small")
        assert s1 != s2
        assert "tiny" in s1.name
        assert "small" in s2.name

    def test_stem_derived_from_transcript(self, tmp_path: Path):
        tp = tmp_path / "my-session.md"
        tp.write_text("")
        result = RetranscribeRunner.sidecar_path(tp, "base")
        assert result.name == "my-session_retranscribe_base.md"

    def test_no_scrub_tag_in_retranscribe_path(self, transcript_path: Path):
        """The new naming must not accidentally use the legacy _scrub_ tag."""
        result = RetranscribeRunner.sidecar_path(transcript_path, "small")
        assert "_scrub_" not in result.name
        assert "_retranscribe_" in result.name


# ---------------------------------------------------------------------------
# Accept / reject — sidecar path consistency
# ---------------------------------------------------------------------------

class TestAcceptRetranscribe:
    def test_accept_replaces_transcript(self, transcript_path: Path):
        sidecar = RetranscribeRunner.sidecar_path(transcript_path, "small")
        sidecar.write_text("# Re-transcribed\n\nBetter text\n")

        result = RetranscribeRunner.accept_scrub(transcript_path, "small")

        assert result == transcript_path
        assert transcript_path.read_text() == "# Re-transcribed\n\nBetter text\n"
        # Sidecar should be gone (moved, not copied)
        assert not sidecar.exists()

    def test_accept_missing_sidecar_raises(self, transcript_path: Path):
        with pytest.raises(FileNotFoundError, match="Sidecar not found"):
            RetranscribeRunner.accept_scrub(transcript_path, "small")


class TestRejectRetranscribe:
    def test_reject_deletes_sidecar(self, transcript_path: Path):
        sidecar = RetranscribeRunner.sidecar_path(transcript_path, "small")
        sidecar.write_text("unwanted")

        RetranscribeRunner.reject_scrub(transcript_path, "small")

        assert not sidecar.exists()

    def test_reject_idempotent(self, transcript_path: Path):
        # Sidecar doesn't exist — should not raise
        RetranscribeRunner.reject_scrub(transcript_path, "small")


# ---------------------------------------------------------------------------
# Backward compat — ScrubRunner still produces legacy _scrub_ naming
# ---------------------------------------------------------------------------

class TestScrubBackwardCompat:
    """The legacy ScrubRunner (imported via the scrub.py shim) must keep
    producing ``_scrub_`` sidecar names so existing callers/tests behave
    unchanged. This locks in the alongside-not-replace nature of the Expand.
    """

    def test_scrub_runner_uses_legacy_scrub_tag(self, transcript_path: Path):
        from meetandread.transcription.scrub import ScrubRunner
        result = ScrubRunner.sidecar_path(transcript_path, "small")
        assert result == transcript_path.parent / (
            "recording-2026-04-26-120000_scrub_small.md"
        )

    def test_scrub_runner_accept_uses_legacy_path(self, transcript_path: Path):
        from meetandread.transcription.scrub import ScrubRunner
        # Create sidecar at the LEGACY path
        sidecar = transcript_path.parent / (
            "recording-2026-04-26-120000_scrub_small.md"
        )
        sidecar.write_text("# Legacy scrub output\n")

        ScrubRunner.accept_scrub(transcript_path, "small")

        assert transcript_path.read_text() == "# Legacy scrub output\n"
        assert not sidecar.exists()

    def test_scrub_runner_and_retranscribe_runner_independent(
        self, transcript_path: Path
    ):
        """Both runners can compute sidecar paths without colliding."""
        from meetandread.transcription.scrub import ScrubRunner
        legacy = ScrubRunner.sidecar_path(transcript_path, "small")
        modern = RetranscribeRunner.sidecar_path(transcript_path, "small")
        assert legacy != modern
        assert legacy.name.endswith("_scrub_small.md")
        assert modern.name.endswith("_retranscribe_small.md")
