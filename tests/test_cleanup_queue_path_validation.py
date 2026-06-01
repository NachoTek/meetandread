"""Tests for CleanupQueue identity_cleanup path containment validation.

Verifies that identity_cleanup paths are validated against allowed roots
(recordings_dir, transcripts_dir) before deletion.  Paths outside allowed
roots are rejected without unlinking; paths inside roots are deleted
normally.  Covers symlink/.. traversal, absolute paths, and missing files.
"""

import json
import os
from pathlib import Path

import pytest

from meetandread.recording.cleanup_queue import (
    CleanupOperation,
    CleanupQueue,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_queue(tmp_path: Path) -> CleanupQueue:
    """Create a CleanupQueue with isolated dirs."""
    rec_dir = tmp_path / "recordings"
    tra_dir = tmp_path / "transcripts"
    rec_dir.mkdir()
    tra_dir.mkdir()
    return CleanupQueue(
        tmp_path / "queue.json",
        recordings_dir=rec_dir,
        transcripts_dir=tra_dir,
    )


def _load_queue_raw(queue_path: Path) -> dict:
    return json.loads(queue_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Tests: inside-root paths are deleted
# ---------------------------------------------------------------------------

class TestInsideRootPaths:
    """Paths within recordings_dir or transcripts_dir are deletable."""

    def test_file_in_recordings_dir_deleted(self, tmp_path: Path) -> None:
        """File under recordings_dir is deleted successfully."""
        q = _make_queue(tmp_path)
        target = q._recordings_dir / "speaker_cache.bin"
        target.write_text("cache-data")

        q.enqueue_identity_cleanup("speaker1", paths=[str(target)])
        result = q.process_pending()

        assert result.processed == 1
        assert not target.exists()

    def test_file_in_transcripts_dir_deleted(self, tmp_path: Path) -> None:
        """File under transcripts_dir is deleted successfully."""
        q = _make_queue(tmp_path)
        target = q._transcripts_dir / "old_transcript.md"
        target.write_text("# transcript")

        q.enqueue_identity_cleanup("old", paths=[str(target)])
        result = q.process_pending()

        assert result.processed == 1
        assert not target.exists()

    def test_nested_path_inside_root_deleted(self, tmp_path: Path) -> None:
        """Nested path inside recordings_dir is deleted."""
        q = _make_queue(tmp_path)
        nested = q._recordings_dir / "speakers" / "john_doe.json"
        nested.parent.mkdir(parents=True)
        nested.write_text("{}")

        q.enqueue_identity_cleanup("john_doe", paths=[str(nested)])
        result = q.process_pending()

        assert result.processed == 1
        assert not nested.exists()

    def test_multiple_valid_paths_all_deleted(self, tmp_path: Path) -> None:
        """Multiple valid paths across both roots are all deleted."""
        q = _make_queue(tmp_path)
        f1 = q._recordings_dir / "a.bin"
        f2 = q._transcripts_dir / "b.md"
        f1.write_text("a")
        f2.write_text("b")

        q.enqueue_identity_cleanup("multi", paths=[str(f1), str(f2)])
        result = q.process_pending()

        assert result.processed == 1
        assert not f1.exists()
        assert not f2.exists()

    def test_missing_inside_root_path_completes(self, tmp_path: Path) -> None:
        """Missing file inside root is treated as already-clean."""
        q = _make_queue(tmp_path)
        missing = q._recordings_dir / "nonexistent.bin"

        q.enqueue_identity_cleanup("ghost", paths=[str(missing)])
        result = q.process_pending()

        assert result.processed == 1
        # Operation completed, not failed
        raw = _load_queue_raw(q.queue_path)
        assert raw["operations"] == []


# ---------------------------------------------------------------------------
# Tests: outside-root paths are rejected
# ---------------------------------------------------------------------------

class TestOutsideRootPaths:
    """Paths outside recordings_dir and transcripts_dir are rejected."""

    def test_absolute_path_outside_roots_rejected(self, tmp_path: Path) -> None:
        """Absolute path outside allowed roots is not deleted."""
        q = _make_queue(tmp_path)
        sensitive = tmp_path / "config" / "secrets.env"
        sensitive.parent.mkdir(parents=True)
        sensitive.write_text("SECRET=12345")

        q.enqueue_identity_cleanup("leak", paths=[str(sensitive)])
        result = q.process_pending()

        assert result.failed == 1
        assert sensitive.exists()  # Not deleted
        assert result.remaining == 1

    def test_dotdot_traversal_rejected(self, tmp_path: Path) -> None:
        """Path traversal via ../.. outside roots is rejected."""
        q = _make_queue(tmp_path)
        # recordings_dir is tmp_path/recordings, so
        # recordings_dir/../../etc/passwd resolves outside
        traversal = str(
            q._recordings_dir / ".." / ".." / "etc" / "passwd"
        )
        q.enqueue_identity_cleanup("traversal", paths=[traversal])
        result = q.process_pending()

        assert result.failed == 1
        assert result.remaining == 1

    def test_system_temp_path_rejected(self, tmp_path: Path) -> None:
        """System temp directory path is rejected."""
        q = _make_queue(tmp_path)
        temp_file = tmp_path / "temp_attack.txt"
        temp_file.write_text("evil")

        q.enqueue_identity_cleanup("temp", paths=[str(temp_file)])
        result = q.process_pending()

        assert result.failed == 1
        assert temp_file.exists()
        assert result.remaining == 1

    def test_home_directory_path_rejected(self, tmp_path: Path) -> None:
        """Home directory path is rejected."""
        q = _make_queue(tmp_path)
        home_path = Path.home() / ".bashrc"

        q.enqueue_identity_cleanup("homedir", paths=[str(home_path)])
        result = q.process_pending()

        assert result.failed == 1
        assert result.remaining == 1

    def test_rejection_records_last_error(self, tmp_path: Path) -> None:
        """Rejected path records last_error on the operation."""
        q = _make_queue(tmp_path)
        outside = tmp_path / "forbidden.txt"
        outside.write_text("no")

        q.enqueue_identity_cleanup("err", paths=[str(outside)])
        result = q.process_pending()

        assert result.failed == 1
        op = q.operations[0]
        assert op.last_error is not None
        assert "outside allowed roots" in op.last_error

    def test_rejection_preserves_file_on_disk(self, tmp_path: Path) -> None:
        """Rejected path does not delete the target file."""
        q = _make_queue(tmp_path)
        outside = tmp_path / "keep_me.txt"
        outside.write_text("important")

        q.enqueue_identity_cleanup("keep", paths=[str(outside)])
        q.process_pending()

        assert outside.exists()
        assert outside.read_text() == "important"

    def test_mixed_valid_and_invalid_paths(self, tmp_path: Path) -> None:
        """Some paths valid, some invalid — partial failure stays pending."""
        q = _make_queue(tmp_path)
        valid = q._recordings_dir / "ok.bin"
        valid.write_text("ok")
        outside = tmp_path / "nope.txt"
        outside.write_text("nope")

        q.enqueue_identity_cleanup("mixed", paths=[str(valid), str(outside)])
        result = q.process_pending()

        # Partial failure: operation stays pending
        assert result.failed == 1
        assert result.remaining == 1
        # Valid path was deleted
        assert not valid.exists()
        # Invalid path was preserved
        assert outside.exists()


# ---------------------------------------------------------------------------
# Tests: symlink handling
# ---------------------------------------------------------------------------

class TestSymlinkHandling:
    """Symlinks pointing outside roots are resolved and rejected."""

    @pytest.mark.skipif(
        os.name == "nt",
        reason="Symlink creation requires admin on Windows",
    )
    def test_symlink_outside_roots_rejected(self, tmp_path: Path) -> None:
        """Symlink inside recordings_dir pointing outside is rejected."""
        q = _make_queue(tmp_path)
        outside = tmp_path / "real_secret.txt"
        outside.write_text("secret")
        symlink = q._recordings_dir / "link_to_secret"
        symlink.symlink_to(outside)

        q.enqueue_identity_cleanup("symlink", paths=[str(symlink)])
        result = q.process_pending()

        # Symlink resolves to outside roots → rejected
        assert result.failed == 1
        assert outside.exists()

    @pytest.mark.skipif(
        os.name == "nt",
        reason="Symlink creation requires admin on Windows",
    )
    def test_symlink_inside_roots_allowed(self, tmp_path: Path) -> None:
        """Symlink inside recordings_dir pointing within roots is allowed."""
        q = _make_queue(tmp_path)
        real = q._recordings_dir / "real_data.bin"
        real.write_text("data")
        link = q._recordings_dir / "link_data"
        link.symlink_to(real)

        q.enqueue_identity_cleanup("goodlink", paths=[str(link)])
        result = q.process_pending()

        # Symlink resolves inside roots → allowed, deleted
        assert result.processed == 1
        assert not link.exists()


# ---------------------------------------------------------------------------
# Tests: path validation edge cases
# ---------------------------------------------------------------------------

class TestPathValidationEdgeCases:
    """Edge cases in path containment validation."""

    def test_empty_path_rejected(self, tmp_path: Path) -> None:
        """Empty string path is rejected."""
        q = _make_queue(tmp_path)
        q.enqueue_identity_cleanup("empty", paths=[""])
        result = q.process_pending()

        assert result.failed == 1

    def test_relative_path_rejected(self, tmp_path: Path) -> None:
        """Relative path is rejected (resolves to cwd, not allowed roots)."""
        q = _make_queue(tmp_path)
        q.enqueue_identity_cleanup("rel", paths=["relative/path.txt"])
        result = q.process_pending()

        # Relative path resolves against cwd, not our roots
        assert result.failed == 1

    def test_operation_stays_retryable_after_rejection(self, tmp_path: Path) -> None:
        """Rejected path operation keeps attempts counter for retry."""
        q = _make_queue(tmp_path)
        outside = tmp_path / "retry_test.txt"
        outside.write_text("x")

        q.enqueue_identity_cleanup("retry", paths=[str(outside)])
        result1 = q.process_pending()
        assert result1.failed == 1

        op = q.operations[0]
        assert op.attempts == 1
        assert op.status == "pending"

        # Second processing attempt still fails
        result2 = q.process_pending()
        assert result2.failed == 1
        assert q.operations[0].attempts == 2
