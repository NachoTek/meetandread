"""Cleanup queue service for deferred file deletion and identity cleanup.

Provides atomic JSON persistence, corrupt queue recovery, enqueue/process
operations, and structured results suitable for UI retry dialogs.

The queue is designed to handle operations that may fail at runtime (e.g.
locked audio files on Windows) and be retried later.
"""

import json
import logging
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from meetandread.recording.management import delete_recording_structured

logger = logging.getLogger(__name__)

# Default queue file path (used only by runtime, tests inject tmp_path)
_DEFAULT_QUEUE_PATH = Path.home() / ".gsd" / "cleanup_queue.json"


@dataclass
class CleanupOperation:
    """A single cleanup operation in the queue.

    Attributes:
        kind: Operation type, e.g. "file_delete" or "identity_cleanup".
        target: What to clean up (stem for recordings, identity name for identities).
        paths: Explicit file paths to delete (for file_delete operations).
        status: "pending" or "completed".
        attempts: Number of times this operation has been attempted.
        last_error: Last error message, if any.
    """
    kind: str
    target: str
    paths: List[str] = field(default_factory=list)
    status: str = "pending"
    attempts: int = 0
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CleanupOperation":
        return cls(
            kind=data.get("kind", "file_delete"),
            target=data.get("target", ""),
            paths=data.get("paths", []),
            status=data.get("status", "pending"),
            attempts=data.get("attempts", 0),
            last_error=data.get("last_error"),
        )


@dataclass
class ProcessResult:
    """Result of processing the cleanup queue.

    Attributes:
        processed: Number of operations that completed successfully.
        failed: Number of operations that failed.
        remaining: Number of pending operations still in the queue.
        details: Per-operation summaries for UI display.
    """
    processed: int = 0
    failed: int = 0
    remaining: int = 0
    details: List[str] = field(default_factory=list)


class CleanupQueue:
    """Persistent queue for deferred cleanup operations.

    Uses atomic JSON writes (write-to-temp then rename) to avoid corruption.
    Corrupt JSON files are detected and reset to an empty queue on load.

    Args:
        queue_path: Path to the queue JSON file. In production this defaults
            to ~/.gsd/cleanup_queue.json, but tests must inject a tmp_path.
        recordings_dir: Optional override for recordings directory.
        transcripts_dir: Optional override for transcripts directory.
    """

    def __init__(
        self,
        queue_path: Path = _DEFAULT_QUEUE_PATH,
        *,
        recordings_dir: Optional[Path] = None,
        transcripts_dir: Optional[Path] = None,
    ):
        self._queue_path = queue_path
        self._recordings_dir = recordings_dir
        self._transcripts_dir = transcripts_dir
        self._operations: List[CleanupOperation] = []
        self._load()

    # -- Persistence -------------------------------------------------------

    def _load(self) -> None:
        """Load queue from disk, recovering from corruption if needed."""
        if not self._queue_path.exists():
            self._operations = []
            return

        try:
            raw = self._queue_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning(
                "Corrupt cleanup queue at %s: %s — resetting to empty",
                self._queue_path, exc,
            )
            self._operations = []
            # Overwrite the corrupt file with a valid empty queue
            self._save()
            return

        if not isinstance(data, dict) or "operations" not in data:
            logger.warning(
                "Invalid cleanup queue format at %s — resetting to empty",
                self._queue_path,
            )
            self._operations = []
            self._save()
            return

        self._operations = [
            CleanupOperation.from_dict(op)
            for op in data["operations"]
            if isinstance(op, dict)
        ]
        logger.debug(
            "Loaded %d operations from cleanup queue", len(self._operations)
        )

    def _save(self) -> None:
        """Atomically persist queue to disk (write-to-temp, then rename)."""
        payload = {
            "operations": [op.to_dict() for op in self._operations],
        }
        self._queue_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Atomic write: temp file in same directory, then rename
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._queue_path.parent),
                prefix=".cleanup_queue_",
                suffix=".tmp",
            )
            try:
                with open(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                # On Windows, need to remove target before rename
                if self._queue_path.exists():
                    self._queue_path.unlink()
                Path(tmp_path).rename(self._queue_path)
            except BaseException:
                # Clean up temp file on any error
                Path(tmp_path).unlink(missing_ok=True)
                raise
        except OSError as exc:
            logger.error("Failed to persist cleanup queue: %s", exc)

    # -- Public API --------------------------------------------------------

    @property
    def queue_path(self) -> Path:
        """Path to the queue file on disk."""
        return self._queue_path

    @property
    def pending_count(self) -> int:
        """Number of pending operations in the queue."""
        return sum(1 for op in self._operations if op.status == "pending")

    @property
    def operations(self) -> List[CleanupOperation]:
        """Read-only view of all operations."""
        return list(self._operations)

    def enqueue_file_deletion(
        self,
        stem: str,
        paths: Optional[List[str]] = None,
    ) -> CleanupOperation:
        """Enqueue a file deletion operation for a recording stem.

        Args:
            stem: Recording stem to delete.
            paths: Optional explicit file paths. If None, paths will be
                resolved at processing time.

        Returns:
            The enqueued CleanupOperation.
        """
        op = CleanupOperation(
            kind="file_delete",
            target=stem,
            paths=paths or [],
        )
        self._operations.append(op)
        self._save()
        logger.info(
            "Enqueued file deletion for stem %s (%d pending)",
            stem, self.pending_count,
        )
        return op

    def enqueue_identity_cleanup(
        self,
        identity_name: str,
        paths: Optional[List[str]] = None,
    ) -> CleanupOperation:
        """Enqueue an identity cleanup operation.

        Args:
            identity_name: Speaker identity name to clean up.
            paths: Optional explicit file paths to remove.

        Returns:
            The enqueued CleanupOperation.
        """
        op = CleanupOperation(
            kind="identity_cleanup",
            target=identity_name,
            paths=paths or [],
        )
        self._operations.append(op)
        self._save()
        logger.info(
            "Enqueued identity cleanup for %s (%d pending)",
            identity_name, self.pending_count,
        )
        return op

    def process_pending(self) -> ProcessResult:
        """Process all pending operations in the queue.

        For file_delete operations, delegates to delete_recording_structured.
        For identity_cleanup, attempts to delete explicit paths.

        Completed operations are removed from the queue. Failed operations
        remain with incremented attempt counts.

        Returns:
            ProcessResult with counts and per-operation summaries.
        """
        result = ProcessResult()
        still_pending: List[CleanupOperation] = []

        for op in self._operations:
            if op.status != "pending":
                continue

            op.attempts += 1

            try:
                if op.kind == "file_delete":
                    summary = self._process_file_delete(op)
                elif op.kind == "identity_cleanup":
                    summary = self._process_identity_cleanup(op)
                else:
                    summary = f"Unknown operation kind: {op.kind}"
                    logger.warning("Unknown cleanup kind: %s", op.kind)
                    op.status = "completed"
                    result.processed += 1
                    result.details.append(summary)
                    continue

                result.details.append(summary)

                # If the operation had no failures, mark completed
                if op.status == "completed":
                    result.processed += 1
                else:
                    still_pending.append(op)
                    result.failed += 1

            except Exception as exc:
                op.last_error = str(exc)
                still_pending.append(op)
                result.failed += 1
                result.details.append(f"Error processing {op.kind} for {op.target}: {exc}")
                logger.error(
                    "Cleanup processing error for %s/%s: %s",
                    op.kind, op.target, exc,
                )

        # Keep completed ops out, retain still-pending
        completed = [op for op in self._operations if op.status == "completed"]
        self._operations = completed + still_pending
        result.remaining = self.pending_count

        self._save()
        logger.info(
            "Cleanup queue processed: %d succeeded, %d failed, %d remaining",
            result.processed, result.failed, result.remaining,
        )
        return result

    def clear_completed(self) -> int:
        """Remove all completed operations from the queue.

        Returns:
            Number of completed operations removed.
        """
        before = len(self._operations)
        self._operations = [
            op for op in self._operations if op.status != "completed"
        ]
        removed = before - len(self._operations)
        if removed:
            self._save()
            logger.info("Cleared %d completed operations from queue", removed)
        return removed

    # -- Internal processors -----------------------------------------------

    def _process_file_delete(self, op: CleanupOperation) -> str:
        """Process a file_delete operation."""
        del_result = delete_recording_structured(
            op.target,
            recordings_dir=self._recordings_dir,
            transcripts_dir=self._transcripts_dir,
        )

        if del_result.all_succeeded:
            op.status = "completed"
            return f"Deleted {del_result.success_count} files for stem {op.target}"
        elif del_result.success_count == 0:
            op.last_error = (
                "Failed to delete any files: "
                + "; ".join(f"{p}: {r}" for p, r in del_result.failed[:3])
            )
            return f"Failed to delete any files for stem {op.target}"
        else:
            # Partial success — keep pending for retry of remaining
            op.last_error = (
                f"Partial delete: {del_result.failure_count} files remain: "
                + "; ".join(f"{p}: {r}" for p, r in del_result.failed[:3])
            )
            return (
                f"Partially deleted {del_result.success_count} files for stem "
                f"{op.target}, {del_result.failure_count} remain"
            )

    def _process_identity_cleanup(self, op: CleanupOperation) -> str:
        """Process an identity_cleanup operation by deleting explicit paths."""
        if not op.paths:
            op.status = "completed"
            return f"No explicit paths for identity cleanup: {op.target}"

        deleted = 0
        failures = []
        for path_str in op.paths:
            path = Path(path_str)
            try:
                if path.exists():
                    path.unlink()
                    deleted += 1
            except OSError as exc:
                failures.append(f"{path_str}: {exc}")

        if not failures:
            op.status = "completed"
            return f"Cleaned up {deleted} paths for identity {op.target}"
        else:
            op.last_error = "; ".join(failures[:3])
            return (
                f"Partially cleaned up identity {op.target}: "
                f"{deleted} deleted, {len(failures)} failed"
            )
