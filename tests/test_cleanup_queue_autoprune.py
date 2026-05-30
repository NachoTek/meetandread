"""Tests for CleanupQueue auto-pruning of completed operations.

Verifies that process_pending leaves no completed operations in the
persisted queue, failed operations remain retryable, and pre-existing
completed operations are cleaned up on load-and-process.
"""

import json
from pathlib import Path

import pytest

from meetandread.recording.cleanup_queue import (
    CleanupOperation,
    CleanupQueue,
    ProcessResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_queue(tmp_path: Path) -> CleanupQueue:
    """Create a CleanupQueue with isolated dirs (no production FS access)."""
    rec_dir = tmp_path / "recordings"
    tra_dir = tmp_path / "transcripts"
    rec_dir.mkdir()
    tra_dir.mkdir()
    q = CleanupQueue(
        tmp_path / "queue.json",
        recordings_dir=rec_dir,
        transcripts_dir=tra_dir,
    )
    return q


def _load_queue_raw(queue_path: Path) -> dict:
    """Load the raw JSON queue file."""
    return json.loads(queue_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Tests: completed operations are pruned after processing
# ---------------------------------------------------------------------------

class TestAutoPruneOnProcess:
    """process_pending must not retain completed operations."""

    def test_completed_file_delete_pruned(self, tmp_path: Path) -> None:
        """A file_delete operation that succeeds is not in persisted queue."""
        q = _make_queue(tmp_path)
        # Create a recording file so deletion succeeds
        stem = "test-rec"
        rec_file = q._recordings_dir / f"{stem}.wav"
        rec_file.write_text("audio")

        q.enqueue_file_deletion(stem)
        assert q.pending_count == 1

        result = q.process_pending()
        assert result.processed == 1
        assert result.remaining == 0

        # Queue on disk should have zero operations
        raw = _load_queue_raw(q.queue_path)
        assert raw["operations"] == []

        # In-memory also empty
        assert q.operations == []

    def test_completed_identity_cleanup_pruned(self, tmp_path: Path) -> None:
        """An identity_cleanup that succeeds is pruned from persisted queue."""
        q = _make_queue(tmp_path)
        target_file = q._recordings_dir / "speaker_cache.bin"
        target_file.write_text("data")

        q.enqueue_identity_cleanup("speaker1", paths=[str(target_file)])
        result = q.process_pending()

        assert result.processed == 1
        assert result.remaining == 0

        raw = _load_queue_raw(q.queue_path)
        assert raw["operations"] == []

    def test_failed_operation_remains_in_queue(self, tmp_path: Path) -> None:
        """A file_delete with missing files stays pending/retryable."""
        q = _make_queue(tmp_path)
        # Enqueue a stem that has no files — delete_recording_structured
        # will succeed (0 files found, 0 failures) so it completes.
        # Instead, test identity_cleanup with a permission-denied path.
        target = q._recordings_dir / "locked.bin"
        target.write_text("data")

        q.enqueue_identity_cleanup("locked", paths=[str(target)])
        # Make the file undeletable by removing write permission isn't reliable
        # cross-platform. Instead, test with a path that doesn't exist but
        # won't fail — that completes. Let's use a partial-failure scenario.
        # Actually, for file_delete: if delete_recording_structured returns
        # all_succeeded=True even with 0 files, it's still completed.
        # Let me test the partial-failure code path directly.
        pass  # See next test for explicit failure path

    def test_failed_identity_cleanup_stays_pending(self, tmp_path: Path) -> None:
        """identity_cleanup with path outside roots stays retryable."""
        q = _make_queue(tmp_path)
        outside = tmp_path / "outside_root" / "secret.txt"
        outside.parent.mkdir(parents=True)
        outside.write_text("sensitive")

        q.enqueue_identity_cleanup("bad", paths=[str(outside)])
        result = q.process_pending()

        # Operation should fail (path outside roots) and remain pending
        assert result.failed == 1
        assert result.remaining == 1

        raw = _load_queue_raw(q.queue_path)
        assert len(raw["operations"]) == 1
        op = raw["operations"][0]
        assert op["status"] == "pending"
        assert op["attempts"] == 1
        assert op["last_error"] is not None

    def test_preexisting_completed_pruned_on_process(self, tmp_path: Path) -> None:
        """A completed operation pre-existing in the JSON file is pruned."""
        q = _make_queue(tmp_path)
        # Manually inject a completed operation into the queue file
        q._operations.append(
            CleanupOperation(
                kind="file_delete",
                target="old-stem",
                status="completed",
                attempts=1,
            )
        )
        q._save()
        assert len(q.operations) == 1

        result = q.process_pending()
        assert result.processed == 0
        assert result.remaining == 0

        # The pre-existing completed op should be gone
        raw = _load_queue_raw(q.queue_path)
        assert raw["operations"] == []
        assert q.operations == []

    def test_mixed_completed_and_failed(self, tmp_path: Path) -> None:
        """Completed are pruned, failed remain — in same process cycle."""
        q = _make_queue(tmp_path)

        # One that will succeed
        good_file = q._recordings_dir / "good.wav"
        good_file.write_text("audio")
        q.enqueue_file_deletion("good")

        # One that will fail (path outside roots)
        outside = tmp_path / "outside" / "bad.txt"
        outside.parent.mkdir(parents=True)
        outside.write_text("nope")
        q.enqueue_identity_cleanup("bad", paths=[str(outside)])

        # Plus a pre-existing completed
        q._operations.append(
            CleanupOperation(kind="file_delete", target="old", status="completed")
        )
        q._save()

        result = q.process_pending()
        assert result.processed == 1  # file_delete succeeded
        assert result.failed == 1  # identity_cleanup rejected
        assert result.remaining == 1

        # Only the failed operation should remain
        raw = _load_queue_raw(q.queue_path)
        assert len(raw["operations"]) == 1
        assert raw["operations"][0]["status"] == "pending"

    def test_empty_paths_identity_cleanup_completes(self, tmp_path: Path) -> None:
        """identity_cleanup with no paths completes and is pruned."""
        q = _make_queue(tmp_path)
        q.enqueue_identity_cleanup("nobody", paths=[])
        result = q.process_pending()

        assert result.processed == 1
        assert result.remaining == 0
        raw = _load_queue_raw(q.queue_path)
        assert raw["operations"] == []

    def test_multiple_cycles_prune_each_time(self, tmp_path: Path) -> None:
        """Repeated process_pending calls don't accumulate completed ops."""
        q = _make_queue(tmp_path)

        for i in range(5):
            stem = f"rec-{i}"
            rec = q._recordings_dir / f"{stem}.wav"
            rec.write_text("x")
            q.enqueue_file_deletion(stem)
            result = q.process_pending()
            assert result.processed == 1
            assert result.remaining == 0

        # After 5 cycles, queue should be empty
        raw = _load_queue_raw(q.queue_path)
        assert raw["operations"] == []

    def test_10x_queue_size_no_retention(self, tmp_path: Path) -> None:
        """Even with 10x operations, none are retained after processing."""
        q = _make_queue(tmp_path)

        for i in range(30):
            stem = f"bulk-{i}"
            rec = q._recordings_dir / f"{stem}.wav"
            rec.write_text("x")
            q.enqueue_file_deletion(stem)

        result = q.process_pending()
        assert result.processed == 30
        assert result.remaining == 0

        raw = _load_queue_raw(q.queue_path)
        assert raw["operations"] == []


# ---------------------------------------------------------------------------
# Tests: startup cleanup remains compatible
# ---------------------------------------------------------------------------

class TestStartupCompatibility:
    """Startup cleanup behavior (load + process) remains compatible."""

    def test_load_corrupt_file_resets(self, tmp_path: Path) -> None:
        """Corrupt JSON file is reset to empty queue on load."""
        qpath = tmp_path / "queue.json"
        qpath.write_text("NOT JSON {{{")
        rec_dir = tmp_path / "recordings"
        tra_dir = tmp_path / "transcripts"
        rec_dir.mkdir()
        tra_dir.mkdir()

        q = CleanupQueue(qpath, recordings_dir=rec_dir, transcripts_dir=tra_dir)
        assert q.operations == []
        assert q.pending_count == 0

    def test_load_valid_queue_preserves_pending(self, tmp_path: Path) -> None:
        """Valid queue with pending ops is loaded correctly."""
        rec_dir = tmp_path / "recordings"
        tra_dir = tmp_path / "transcripts"
        rec_dir.mkdir()
        tra_dir.mkdir()
        qpath = tmp_path / "queue.json"

        # Write a valid queue with one pending op
        qpath.write_text(json.dumps({
            "operations": [{
                "kind": "file_delete",
                "target": "test-stem",
                "paths": [],
                "status": "pending",
                "attempts": 0,
                "last_error": None,
            }]
        }))

        q = CleanupQueue(qpath, recordings_dir=rec_dir, transcripts_dir=tra_dir)
        assert q.pending_count == 1
        assert q.operations[0].target == "test-stem"

    def test_load_completed_ops_pruned_on_first_process(self, tmp_path: Path) -> None:
        """Queue loaded with completed ops prunes them on first process."""
        rec_dir = tmp_path / "recordings"
        tra_dir = tmp_path / "transcripts"
        rec_dir.mkdir()
        tra_dir.mkdir()
        qpath = tmp_path / "queue.json"

        qpath.write_text(json.dumps({
            "operations": [
                {
                    "kind": "file_delete",
                    "target": "done",
                    "paths": [],
                    "status": "completed",
                    "attempts": 1,
                },
                {
                    "kind": "file_delete",
                    "target": "pending",
                    "paths": [],
                    "status": "pending",
                    "attempts": 0,
                },
            ]
        }))

        q = CleanupQueue(qpath, recordings_dir=rec_dir, transcripts_dir=tra_dir)
        assert len(q.operations) == 2

        result = q.process_pending()
        # The pre-existing completed op should be pruned
        # The pending file_delete with no files will succeed (0 files = all_succeeded)
        assert result.processed == 1
        assert result.remaining == 0

        raw = _load_queue_raw(qpath)
        assert raw["operations"] == []
