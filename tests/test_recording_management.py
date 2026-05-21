"""Tests for recording file management and cleanup queue services.

Covers: enumerate, rename (success/conflict/rollback), delete (structured),
cleanup queue persistence/recovery/processing.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from meetandread.recording.management import (
    DeletionResult,
    RenameResult,
    _validate_stem,
    delete_recording,
    delete_recording_structured,
    enumerate_recording_files,
    rename_recording,
)
from meetandread.recording.cleanup_queue import (
    CleanupOperation,
    CleanupQueue,
    ProcessResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def recording_dirs(tmp_path):
    """Create recordings/ and transcripts/ directories under tmp_path."""
    recordings = tmp_path / "recordings"
    transcripts = tmp_path / "transcripts"
    recordings.mkdir()
    transcripts.mkdir()
    return recordings, transcripts


@pytest.fixture(autouse=True)
def patch_dirs(monkeypatch, recording_dirs):
    """Redirect path resolution to temp directories."""
    recordings_dir, transcripts_dir = recording_dirs

    monkeypatch.setattr(
        "meetandread.recording.management.get_recordings_dir",
        lambda: recordings_dir,
    )
    monkeypatch.setattr(
        "meetandread.recording.management.get_transcripts_dir",
        lambda: transcripts_dir,
    )


def _create_recording(recording_dirs, stem, with_sidecar=False, with_pcm=False):
    """Helper to create a set of recording files on disk."""
    rec_dir, tra_dir = recording_dirs
    created = []
    wav = rec_dir / f"{stem}.wav"
    wav.write_text("audio")
    created.append(wav)

    md = tra_dir / f"{stem}.md"
    md.write_text("# transcript")
    created.append(md)

    if with_pcm:
        pcm = rec_dir / f"{stem}.pcm.part"
        pcm.write_bytes(b"\x00\x01")
        created.append(pcm)
        meta = rec_dir / f"{stem}.pcm.part.json"
        meta.write_text("{}")
        created.append(meta)

    if with_sidecar:
        sc = tra_dir / f"{stem}_scrub_v1.md"
        sc.write_text("# scrub")
        created.append(sc)

    return created


# ---------------------------------------------------------------------------
# Stem validation
# ---------------------------------------------------------------------------

class TestStemValidation:
    def test_valid_stem(self):
        _validate_stem("recording-2026-01-01-120000")

    def test_valid_stem_with_underscores(self):
        _validate_stem("my_recording_001")

    def test_valid_stem_with_dots(self):
        _validate_stem("recording.v2")

    def test_empty_stem_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            _validate_stem("")

    def test_path_traversal_raises(self):
        with pytest.raises(ValueError, match="Invalid stem"):
            _validate_stem("../etc/passwd")

    def test_slash_raises(self):
        with pytest.raises(ValueError, match="Invalid stem"):
            _validate_stem("sub/ recording")

    def test_backslash_raises(self):
        with pytest.raises(ValueError, match="Invalid stem"):
            _validate_stem("sub\\recording")

    def test_null_raises(self):
        with pytest.raises(ValueError, match="Invalid stem"):
            _validate_stem("rec\x00ording")


# ---------------------------------------------------------------------------
# Enumerate recording files (with optional dir overrides)
# ---------------------------------------------------------------------------

class TestEnumerateRecordingFiles:
    def test_enumerate_finds_md_and_wav(self, recording_dirs):
        recordings_dir, transcripts_dir = recording_dirs
        stem = "recording-2026-04-01-120000"
        (recordings_dir / f"{stem}.wav").write_text("audio")
        (transcripts_dir / f"{stem}.md").write_text("# transcript")

        found = enumerate_recording_files(stem)
        names = {p.name for p in found}
        assert f"{stem}.wav" in names
        assert f"{stem}.md" in names

    def test_enumerate_with_dir_overrides(self, tmp_path):
        """When dir overrides are provided, they are used instead of defaults."""
        custom_rec = tmp_path / "custom_rec"
        custom_tra = tmp_path / "custom_tra"
        custom_rec.mkdir()
        custom_tra.mkdir()
        stem = "test-stem"
        (custom_rec / f"{stem}.wav").write_text("a")
        (custom_tra / f"{stem}.md").write_text("b")

        found = enumerate_recording_files(
            stem, recordings_dir=custom_rec, transcripts_dir=custom_tra
        )
        names = {p.name for p in found}
        assert f"{stem}.wav" in names
        assert f"{stem}.md" in names

    def test_enumerate_finds_pcm_parts(self, recording_dirs):
        recordings_dir, _ = recording_dirs
        stem = "rec-001"
        (recordings_dir / f"{stem}.pcm.part").write_bytes(b"\x00")
        (recordings_dir / f"{stem}.pcm.part.json").write_text("{}")

        found = enumerate_recording_files(stem)
        names = {p.name for p in found}
        assert f"{stem}.pcm.part" in names
        assert f"{stem}.pcm.part.json" in names

    def test_enumerate_finds_sidecars(self, recording_dirs):
        _, transcripts_dir = recording_dirs
        stem = "rec-sidecar"
        (transcripts_dir / f"{stem}.md").write_text("t")
        (transcripts_dir / f"{stem}_scrub_v1.md").write_text("s1")
        (transcripts_dir / f"{stem}_scrub_v2.md").write_text("s2")

        found = enumerate_recording_files(stem)
        names = {p.name for p in found}
        assert f"{stem}_scrub_v1.md" in names
        assert f"{stem}_scrub_v2.md" in names

    def test_enumerate_skips_missing(self, recording_dirs):
        recordings_dir, _ = recording_dirs
        stem = "only-wav"
        (recordings_dir / f"{stem}.wav").write_text("a")

        found = enumerate_recording_files(stem)
        assert len(found) == 1
        assert found[0].name == f"{stem}.wav"


# ---------------------------------------------------------------------------
# Rename recording
# ---------------------------------------------------------------------------

class TestRenameRecording:
    def test_rename_success_wav_and_md(self, recording_dirs):
        rec_dir, tra_dir = recording_dirs
        old_stem = "old-name"
        new_stem = "new-name"
        _create_recording(recording_dirs, old_stem)

        result = rename_recording(old_stem, new_stem)

        assert len(result.renamed) == 2
        assert result.failed == []
        assert result.rolled_back == []
        assert not (rec_dir / f"{old_stem}.wav").exists()
        assert not (tra_dir / f"{old_stem}.md").exists()
        assert (rec_dir / f"{new_stem}.wav").exists()
        assert (tra_dir / f"{new_stem}.md").exists()

    def test_rename_with_pcm_parts(self, recording_dirs):
        rec_dir, _ = recording_dirs
        old_stem = "with-pcm"
        new_stem = "renamed-pcm"
        _create_recording(recording_dirs, old_stem, with_pcm=True)

        result = rename_recording(old_stem, new_stem)

        assert len(result.renamed) == 4  # wav + pcm.part + pcm.part.json + md
        assert (rec_dir / f"{new_stem}.pcm.part").exists()
        assert (rec_dir / f"{new_stem}.pcm.part.json").exists()

    def test_rename_with_scrub_sidecars(self, recording_dirs):
        _, tra_dir = recording_dirs
        old_stem = "with-scrub"
        new_stem = "scrubbed"
        _create_recording(recording_dirs, old_stem, with_sidecar=True)

        result = rename_recording(old_stem, new_stem)

        assert len(result.renamed) == 3  # wav + md + scrub sidecar
        assert (tra_dir / f"{new_stem}_scrub_v1.md").exists()

    def test_rename_target_conflict_aborts(self, recording_dirs):
        rec_dir, tra_dir = recording_dirs
        old_stem = "original"
        new_stem = "collision"
        _create_recording(recording_dirs, old_stem)
        # Create target that already exists
        (rec_dir / f"{new_stem}.wav").write_text("blocking")

        result = rename_recording(old_stem, new_stem)

        assert len(result.failed) > 0
        assert result.renamed == []
        # Original should be untouched
        assert (rec_dir / f"{old_stem}.wav").exists()

    def test_rename_invalid_old_stem_raises(self, recording_dirs):
        with pytest.raises(ValueError):
            rename_recording("../bad", "good")

    def test_rename_invalid_new_stem_raises(self, recording_dirs):
        with pytest.raises(ValueError):
            rename_recording("good", "bad/path")

    def test_rename_no_files_found(self, recording_dirs):
        result = rename_recording("nonexistent-stem", "new-name")
        assert result.renamed == []
        assert result.failed == []

    def test_rename_rollback_on_failure(self, recording_dirs):
        """Inject a failure mid-rename and verify rollback."""
        rec_dir, tra_dir = recording_dirs
        old_stem = "rollback-test"
        new_stem = "rollback-target"
        _create_recording(recording_dirs, old_stem, with_pcm=True)

        # We'll inject a failure by making one target file read-only after
        # the first rename succeeds. We use a monkeypatch approach instead.
        original_rename = Path.rename

        call_count = 0

        def failing_rename(self_path, target):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                # Fail on the second rename
                raise OSError("Injected failure for test")
            return original_rename(self_path, target)

        with patch.object(Path, "rename", failing_rename):
            result = rename_recording(old_stem, new_stem)

        assert len(result.failed) > 0
        assert result.rolled_back_successfully is True
        # After rollback, original files should exist
        assert (rec_dir / f"{old_stem}.wav").exists()

    def test_rename_with_dir_overrides(self, tmp_path):
        """Dir overrides are used for rename operations."""
        custom_rec = tmp_path / "rec"
        custom_tra = tmp_path / "tra"
        custom_rec.mkdir()
        custom_tra.mkdir()
        stem = "override-test"
        (custom_rec / f"{stem}.wav").write_text("a")
        (custom_tra / f"{stem}.md").write_text("b")

        result = rename_recording(
            stem, "renamed",
            recordings_dir=custom_rec,
            transcripts_dir=custom_tra,
        )
        assert len(result.renamed) == 2
        assert (custom_rec / "renamed.wav").exists()


# ---------------------------------------------------------------------------
# Delete recording (backward compat)
# ---------------------------------------------------------------------------

class TestDeleteRecording:
    def test_delete_removes_all(self, recording_dirs):
        rec_dir, tra_dir = recording_dirs
        stem = "del-test"
        _create_recording(recording_dirs, stem)

        count, deleted = delete_recording(stem)

        assert count == 2
        assert not (rec_dir / f"{stem}.wav").exists()
        assert not (tra_dir / f"{stem}.md").exists()

    def test_delete_with_dir_overrides(self, tmp_path):
        custom_rec = tmp_path / "r"
        custom_tra = tmp_path / "t"
        custom_rec.mkdir()
        custom_tra.mkdir()
        stem = "custom-del"
        (custom_rec / f"{stem}.wav").write_text("a")
        (custom_tra / f"{stem}.md").write_text("b")

        count, deleted = delete_recording(
            stem, recordings_dir=custom_rec, transcripts_dir=custom_tra
        )
        assert count == 2


# ---------------------------------------------------------------------------
# Delete recording structured
# ---------------------------------------------------------------------------

class TestDeleteRecordingStructured:
    def test_all_succeed(self, recording_dirs):
        rec_dir, _ = recording_dirs
        stem = "ok-del"
        _create_recording(recording_dirs, stem)

        result = delete_recording_structured(stem)

        assert isinstance(result, DeletionResult)
        assert result.all_succeeded
        assert result.success_count == 2
        assert result.failure_count == 0

    def test_partial_failure(self, recording_dirs):
        """One file is locked — structured result reports partial failure."""
        rec_dir, tra_dir = recording_dirs
        stem = "partial-del"
        wav = rec_dir / f"{stem}.wav"
        wav.write_text("audio")
        md = tra_dir / f"{stem}.md"
        md.write_text("transcript")

        # Make md file undeletable by opening it exclusively (Windows)
        # Instead, mock unlink for the md file
        original_unlink = Path.unlink

        def selective_unlink(self, *args, **kwargs):
            if self.name == f"{stem}.md":
                raise OSError("File is locked (test)")
            return original_unlink(self, *args, **kwargs)

        with patch.object(Path, "unlink", selective_unlink):
            result = delete_recording_structured(stem)

        assert not result.all_succeeded
        assert result.success_count == 1  # wav deleted
        assert result.failure_count == 1  # md failed
        assert len(result.failed) == 1
        assert "locked" in result.failed[0][1]

    def test_with_dir_overrides(self, tmp_path):
        custom_rec = tmp_path / "r"
        custom_tra = tmp_path / "t"
        custom_rec.mkdir()
        custom_tra.mkdir()
        stem = "struct-override"
        (custom_rec / f"{stem}.wav").write_text("a")

        result = delete_recording_structured(
            stem, recordings_dir=custom_rec, transcripts_dir=custom_tra
        )
        assert result.success_count == 1


# ---------------------------------------------------------------------------
# CleanupOperation model
# ---------------------------------------------------------------------------

class TestCleanupOperation:
    def test_to_dict_round_trip(self):
        op = CleanupOperation(
            kind="file_delete", target="stem-1",
            paths=["/a.wav"], status="pending", attempts=2,
            last_error="locked",
        )
        d = op.to_dict()
        restored = CleanupOperation.from_dict(d)
        assert restored.kind == op.kind
        assert restored.target == op.target
        assert restored.paths == op.paths
        assert restored.status == op.status
        assert restored.attempts == op.attempts
        assert restored.last_error == op.last_error

    def test_from_dict_defaults(self):
        op = CleanupOperation.from_dict({})
        assert op.kind == "file_delete"
        assert op.target == ""
        assert op.paths == []
        assert op.status == "pending"
        assert op.attempts == 0
        assert op.last_error is None


# ---------------------------------------------------------------------------
# CleanupQueue
# ---------------------------------------------------------------------------

class TestCleanupQueue:
    def test_enqueue_and_persist(self, tmp_path):
        queue_file = tmp_path / "queue.json"
        q = CleanupQueue(queue_file)

        op = q.enqueue_file_deletion("rec-001")

        assert op.kind == "file_delete"
        assert op.target == "rec-001"
        assert op.status == "pending"
        assert q.pending_count == 1

        # Verify persisted
        data = json.loads(queue_file.read_text())
        assert len(data["operations"]) == 1
        assert data["operations"][0]["target"] == "rec-001"

    def test_load_existing_queue(self, tmp_path):
        queue_file = tmp_path / "queue.json"
        # Pre-populate
        queue_file.write_text(json.dumps({
            "operations": [
                {"kind": "file_delete", "target": "existing", "paths": [],
                 "status": "pending", "attempts": 1, "last_error": None},
            ]
        }))

        q = CleanupQueue(queue_file)
        assert q.pending_count == 1
        assert q.operations[0].target == "existing"

    def test_corrupt_json_resets_queue(self, tmp_path):
        queue_file = tmp_path / "queue.json"
        queue_file.write_text("{not valid json!!!")

        q = CleanupQueue(queue_file)
        assert q.pending_count == 0

        # The file should now be valid JSON
        data = json.loads(queue_file.read_text())
        assert data == {"operations": []}

    def test_corrupt_unicode_resets_queue(self, tmp_path):
        queue_file = tmp_path / "queue.json"
        queue_file.write_bytes(b"\xff\xfe\x00\x00bad")

        q = CleanupQueue(queue_file)
        assert q.pending_count == 0

    def test_invalid_format_resets_queue(self, tmp_path):
        queue_file = tmp_path / "queue.json"
        queue_file.write_text(json.dumps({"wrong_key": []}))

        q = CleanupQueue(queue_file)
        assert q.pending_count == 0

    def test_process_file_delete_success(self, tmp_path, recording_dirs):
        queue_file = tmp_path / "queue.json"
        rec_dir, tra_dir = recording_dirs
        stem = "queue-del"
        (rec_dir / f"{stem}.wav").write_text("a")
        (tra_dir / f"{stem}.md").write_text("t")

        q = CleanupQueue(
            queue_file,
            recordings_dir=rec_dir,
            transcripts_dir=tra_dir,
        )
        q.enqueue_file_deletion(stem)

        result = q.process_pending()

        assert result.processed == 1
        assert result.failed == 0
        assert result.remaining == 0
        assert not (rec_dir / f"{stem}.wav").exists()

    def test_process_file_delete_partial_failure(self, tmp_path, recording_dirs):
        queue_file = tmp_path / "queue.json"
        rec_dir, tra_dir = recording_dirs
        stem = "partial-queue"
        (rec_dir / f"{stem}.wav").write_text("a")
        md = tra_dir / f"{stem}.md"
        md.write_text("t")

        original_unlink = Path.unlink

        def selective_fail(self, *args, **kwargs):
            if self.name == f"{stem}.md":
                raise OSError("locked")
            return original_unlink(self, *args, **kwargs)

        q = CleanupQueue(
            queue_file,
            recordings_dir=rec_dir,
            transcripts_dir=tra_dir,
        )
        q.enqueue_file_deletion(stem)

        with patch.object(Path, "unlink", selective_fail):
            result = q.process_pending()

        # Partial failure: wav deleted but md failed → operation stays pending
        assert result.processed == 0
        assert result.failed == 1
        # The file_delete operation should still be pending for retry
        assert q.pending_count == 1

    def test_process_identity_cleanup_success(self, tmp_path):
        queue_file = tmp_path / "queue.json"
        target_file = tmp_path / "speaker_data.json"
        target_file.write_text("{}")

        q = CleanupQueue(queue_file)
        q.enqueue_identity_cleanup(
            "Speaker1", paths=[str(target_file)]
        )

        result = q.process_pending()

        assert result.processed == 1
        assert not target_file.exists()

    def test_process_identity_cleanup_no_paths(self, tmp_path):
        queue_file = tmp_path / "queue.json"
        q = CleanupQueue(queue_file)
        q.enqueue_identity_cleanup("Speaker1")

        result = q.process_pending()

        assert result.processed == 1
        assert "No explicit paths" in result.details[0]

    def test_multiple_operations_sequential(self, tmp_path, recording_dirs):
        queue_file = tmp_path / "queue.json"
        rec_dir, tra_dir = recording_dirs

        # Create two recordings
        for stem in ["rec-a", "rec-b"]:
            (rec_dir / f"{stem}.wav").write_text("a")
            (tra_dir / f"{stem}.md").write_text("t")

        q = CleanupQueue(
            queue_file,
            recordings_dir=rec_dir,
            transcripts_dir=tra_dir,
        )
        q.enqueue_file_deletion("rec-a")
        q.enqueue_file_deletion("rec-b")

        assert q.pending_count == 2

        result = q.process_pending()

        assert result.processed == 2
        assert result.failed == 0
        assert result.remaining == 0

    def test_clear_completed(self, tmp_path):
        queue_file = tmp_path / "queue.json"
        q = CleanupQueue(queue_file)
        q.enqueue_file_deletion("done-stem")
        q.enqueue_file_deletion("pending-stem")

        # Manually mark one as completed
        q._operations[0].status = "completed"
        q._save()

        removed = q.clear_completed()
        assert removed == 1
        assert q.pending_count == 1

    def test_atomic_write_no_partial_files(self, tmp_path):
        """Verify that save doesn't leave temp files around."""
        queue_file = tmp_path / "queue.json"
        q = CleanupQueue(queue_file)
        q.enqueue_file_deletion("stem-1")

        # No temp files should exist
        temp_files = list(tmp_path.glob(".cleanup_queue_*"))
        assert len(temp_files) == 0

        # Queue file itself should exist
        assert queue_file.exists()

    def test_attempt_count_increments_on_failure(self, tmp_path, recording_dirs):
        queue_file = tmp_path / "queue.json"
        rec_dir, tra_dir = recording_dirs
        stem = "retry-stem"
        md = tra_dir / f"{stem}.md"
        md.write_text("t")

        original_unlink = Path.unlink

        def always_fail(self, *args, **kwargs):
            if self.name == f"{stem}.md":
                raise OSError("locked")
            return original_unlink(self, *args, **kwargs)

        q = CleanupQueue(
            queue_file,
            recordings_dir=rec_dir,
            transcripts_dir=tra_dir,
        )
        q.enqueue_file_deletion(stem)

        with patch.object(Path, "unlink", always_fail):
            q.process_pending()

        # Operation should have attempts = 1
        assert q.operations[0].attempts == 1

        with patch.object(Path, "unlink", always_fail):
            q.process_pending()

        assert q.operations[0].attempts == 2

    def test_empty_queue_process(self, tmp_path):
        queue_file = tmp_path / "queue.json"
        q = CleanupQueue(queue_file)
        result = q.process_pending()

        assert result.processed == 0
        assert result.failed == 0
        assert result.remaining == 0


# ---------------------------------------------------------------------------
# Integration: rename then enumerate
# ---------------------------------------------------------------------------

class TestRenameThenEnumerate:
    def test_after_rename_enumerate_finds_new_stem(self, recording_dirs):
        rec_dir, tra_dir = recording_dirs
        old_stem = "before"
        new_stem = "after"
        _create_recording(recording_dirs, old_stem, with_sidecar=True)

        rename_recording(old_stem, new_stem)

        # Old stem should find nothing
        old_files = enumerate_recording_files(old_stem)
        assert len(old_files) == 0

        # New stem should find everything
        new_files = enumerate_recording_files(new_stem)
        names = {p.name for p in new_files}
        assert f"{new_stem}.wav" in names
        assert f"{new_stem}.md" in names
        assert f"{new_stem}_scrub_v1.md" in names
