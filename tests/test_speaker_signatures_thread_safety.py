"""Thread-safety tests for VoiceSignatureStore concurrent write serialization.

Validates that concurrent save, update, and delete operations against a
shared on-disk SQLite database complete without OperationalError and
produce consistent final state.  Also covers negative cases: duplicate
names, missing-name updates/deletes, and close/write ordering.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pytest

from meetandread.speaker.signatures import VoiceSignatureStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_embedding(dim: int = 256, seed: int = 0) -> np.ndarray:
    """Return a deterministic unit-norm float32 embedding."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def file_store(tmp_path: Path) -> VoiceSignatureStore:
    """File-backed store for thread-safety tests (WAL mode)."""
    db = tmp_path / "thread_safety_test.db"
    with VoiceSignatureStore(str(db)) as s:
        yield s


# ===========================================================================
# Concurrent save tests
# ===========================================================================

class TestConcurrentSave:
    """Multiple threads calling save_signature simultaneously."""

    def test_concurrent_saves_distinct_names(self, file_store: VoiceSignatureStore) -> None:
        """10 threads save distinct speakers through a shared instance; all appear in final profiles."""
        num_threads = 10
        errors: list[Exception] = []

        def worker(idx: int) -> None:
            try:
                file_store.save_signature(
                    f"Speaker_{idx}", _random_embedding(seed=idx)
                )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors during concurrent saves: {errors}"
        profiles = file_store.load_signatures()
        names = {p.name for p in profiles}
        assert names == {f"Speaker_{i}" for i in range(num_threads)}

    def test_concurrent_saves_same_name(self, file_store: VoiceSignatureStore) -> None:
        """10 threads upsert the same name through a shared instance; exactly one row survives."""
        num_threads = 10
        errors: list[Exception] = []

        def worker(idx: int) -> None:
            try:
                file_store.save_signature(
                    "SameSpeaker", _random_embedding(seed=idx), idx + 1
                )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors during concurrent upserts: {errors}"
        profiles = file_store.load_signatures()
        assert len(profiles) == 1, "Exactly one row should survive for the same name"
        assert profiles[0].name == "SameSpeaker"

    def test_shared_instance_concurrent_saves(self, file_store: VoiceSignatureStore) -> None:
        """Concurrent saves through a single shared instance."""
        num_threads = 10
        errors: list[Exception] = []

        def worker(idx: int) -> None:
            try:
                file_store.save_signature(
                    f"Shared_{idx}", _random_embedding(seed=idx)
                )
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors: {errors}"
        profiles = file_store.load_signatures()
        assert len(profiles) == num_threads


# ===========================================================================
# Concurrent update tests
# ===========================================================================

class TestConcurrentUpdate:
    """Multiple threads calling update_signature simultaneously."""

    def test_concurrent_updates_same_speaker(self, file_store: VoiceSignatureStore) -> None:
        """10 threads update the same speaker; num_samples increments correctly."""
        file_store.save_signature("Target", _random_embedding(seed=0), 1)
        num_updates = 10
        errors: list[Exception] = []

        def worker(idx: int) -> None:
            try:
                file_store.update_signature("Target", _random_embedding(seed=100 + idx))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_updates)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors during concurrent updates: {errors}"
        profiles = file_store.load_signatures()
        assert len(profiles) == 1
        assert profiles[0].num_samples == 1 + num_updates

    def test_update_missing_name_returns_false(self, file_store: VoiceSignatureStore) -> None:
        """Concurrent updates to non-existent speakers all return False."""
        results: list[bool] = []
        lock = threading.Lock()

        def worker(idx: int) -> None:
            r = file_store.update_signature(f"Ghost_{idx}", _random_embedding(seed=idx))
            with lock:
                results.append(r)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert all(r is False for r in results)


# ===========================================================================
# Concurrent delete tests
# ===========================================================================

class TestConcurrentDelete:
    """Multiple threads calling delete_signature simultaneously."""

    def test_concurrent_delete_same_name(self, file_store: VoiceSignatureStore) -> None:
        """10 threads delete the same name through a shared instance; only one returns True."""
        file_store.save_signature("Victim", _random_embedding(seed=0))

        results: list[bool] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def worker() -> None:
            try:
                r = file_store.delete_signature("Victim")
                with lock:
                    results.append(r)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors during concurrent deletes: {errors}"
        true_count = sum(1 for r in results if r is True)
        assert true_count == 1, f"Expected exactly 1 True from 10 deletes, got {true_count}"

    def test_delete_missing_name_returns_false(self, file_store: VoiceSignatureStore) -> None:
        """Deleting a non-existent name from multiple threads returns False."""
        results: list[bool] = []
        lock = threading.Lock()

        def worker() -> None:
            r = file_store.delete_signature("Nobody")
            with lock:
                results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert all(r is False for r in results)


# ===========================================================================
# Mixed concurrent operations
# ===========================================================================

class TestMixedConcurrentOperations:
    """Interleaved save/update/delete/read under contention."""

    def test_mixed_operations_no_crash(self, file_store: VoiceSignatureStore) -> None:
        """Mixed save, update, delete, and load from many threads."""
        # Pre-seed some entries
        for i in range(5):
            file_store.save_signature(f"Base_{i}", _random_embedding(seed=i))

        errors: list[Exception] = []

        def writer(idx: int) -> None:
            try:
                name = f"Mixed_{idx}"
                file_store.save_signature(name, _random_embedding(seed=idx))
                file_store.update_signature(name, _random_embedding(seed=idx + 100))
                if idx % 2 == 0:
                    file_store.delete_signature(name)
            except Exception as exc:
                errors.append(exc)

        def reader() -> None:
            try:
                file_store.load_signatures()
                file_store.find_match(_random_embedding(seed=999))
            except Exception as exc:
                errors.append(exc)

        threads = []
        for i in range(20):
            threads.append(threading.Thread(target=writer, args=(i,)))
        for _ in range(10):
            threads.append(threading.Thread(target=reader))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert not errors, f"Errors during mixed operations: {errors}"

    def test_shared_instance_stress_10x_writes(self, file_store: VoiceSignatureStore) -> None:
        """Stress test: 10x write amplification on a single shared instance.

        At 10x writes, operations serialize and latency increases, but
        there must be no DB corruption or connection races.
        """
        num_writers = 10
        writes_per_thread = 10
        errors: list[Exception] = []

        def writer(thread_idx: int) -> None:
            for j in range(writes_per_thread):
                try:
                    name = f"Stress_{thread_idx}_{j}"
                    file_store.save_signature(name, _random_embedding(seed=thread_idx * 100 + j))
                except Exception as exc:
                    errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(num_writers)]
        start = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        elapsed = time.monotonic() - start

        assert not errors, f"Errors during stress test: {errors}"
        profiles = file_store.load_signatures()
        assert len(profiles) == num_writers * writes_per_thread
        # Serialization should complete within a reasonable time
        assert elapsed < 30, f"Stress test took {elapsed:.1f}s — possible deadlock"


# ===========================================================================
# Close/write ordering
# ===========================================================================

class TestCloseWriteOrdering:
    """Negative tests: close during or before writes."""

    def test_close_then_write_raises(self, tmp_path: Path) -> None:
        """Writing to a closed store raises RuntimeError (Store is closed)."""
        db = tmp_path / "closed_store.db"
        store = VoiceSignatureStore(str(db))
        store.close()

        with pytest.raises(RuntimeError, match="VoiceSignatureStore is closed"):
            store.save_signature("Ghost", _random_embedding(seed=0))

    def test_concurrent_close_and_write(self, tmp_path: Path) -> None:
        """Close and write racing from different threads.

        The write should either succeed (lock acquired before close) or
        raise RuntimeError (lock acquired after close). No OperationalError
        or segfault should occur.
        """
        db = tmp_path / "race_close.db"
        store = VoiceSignatureStore(str(db))
        store.save_signature("Existing", _random_embedding(seed=0))

        errors: list[Exception] = []

        def close_worker() -> None:
            try:
                store.close()
            except Exception as exc:
                errors.append(exc)

        def write_worker() -> None:
            try:
                store.save_signature("Racer", _random_embedding(seed=1))
            except (RuntimeError, AssertionError):
                pass  # Expected if close won the race
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=close_worker)
        t2 = threading.Thread(target=write_worker)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        # Only unexpected errors are a problem
        unexpected = [
            e for e in errors
            if not isinstance(e, AssertionError)
        ]
        assert not unexpected, f"Unexpected errors: {unexpected}"

    def test_double_close_safe(self, tmp_path: Path) -> None:
        """Calling close() twice does not raise."""
        db = tmp_path / "double_close.db"
        store = VoiceSignatureStore(str(db))
        store.close()
        store.close()  # Should be a no-op


# ===========================================================================
# Read concurrency during writes
# ===========================================================================

class TestReadDuringWrites:
    """Reads should not block or crash while writes are serialized."""

    def test_reads_during_concurrent_saves(self, file_store: VoiceSignatureStore) -> None:
        """Concurrent reads and writes through the same shared store instance."""
        num_writers = 5
        num_readers = 1
        errors: list[Exception] = []

        # Pre-seed
        file_store.save_signature("Base", _random_embedding(seed=0))

        barrier = threading.Barrier(num_writers + num_readers)

        def writer(idx: int) -> None:
            barrier.wait(timeout=5)
            try:
                file_store.save_signature(f"Writer_{idx}", _random_embedding(seed=idx))
            except Exception as exc:
                errors.append(exc)

        def reader() -> None:
            barrier.wait(timeout=5)
            try:
                for _ in range(20):
                    file_store.load_signatures()
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=writer, args=(i,)) for i in range(num_writers)
        ] + [threading.Thread(target=reader)]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert not errors, f"Errors during concurrent reads/writes: {errors}"

        profiles = file_store.load_signatures()
        # Base + 5 writers = 6
        assert len(profiles) == 1 + num_writers


# ===========================================================================
# Consistency verification
# ===========================================================================

class TestConsistencyAfterConcurrency:
    """Verify data integrity after concurrent operations."""

    def test_final_profiles_consistent_after_save_update_delete(
        self, file_store: VoiceSignatureStore
    ) -> None:
        """After concurrent save/update/delete, final state is self-consistent."""
        # Phase 1: Concurrent saves
        def save_worker(idx: int) -> None:
            file_store.save_signature(f"Con_{idx}", _random_embedding(seed=idx))

        threads = [threading.Thread(target=save_worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # Phase 2: Concurrent updates
        def update_worker(idx: int) -> None:
            file_store.update_signature(f"Con_{idx}", _random_embedding(seed=idx + 50))

        threads = [threading.Thread(target=update_worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # Phase 3: Delete half
        def delete_worker(idx: int) -> None:
            file_store.delete_signature(f"Con_{idx}")

        threads = [threading.Thread(target=delete_worker, args=(i,)) for i in range(0, 10, 2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # Verify
        profiles = file_store.load_signatures()
        remaining_names = {p.name for p in profiles}
        expected = {f"Con_{i}" for i in range(1, 10, 2)}
        assert remaining_names == expected

        # Updated profiles should have num_samples == 2 (initial 1 + one update)
        for p in profiles:
            assert p.num_samples == 2, f"{p.name} should have 2 samples, got {p.num_samples}"
            assert p.embedding.shape == (256,), f"Embedding dim mismatch for {p.name}"

    def test_embedding_not_corrupted_after_concurrent_updates(
        self, file_store: VoiceSignatureStore
    ) -> None:
        """Embeddings remain valid float32 vectors after concurrent updates."""
        file_store.save_signature("Integ", _random_embedding(seed=0), 1)

        def updater(idx: int) -> None:
            file_store.update_signature("Integ", _random_embedding(seed=100 + idx))

        threads = [threading.Thread(target=updater, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        profiles = file_store.load_signatures()
        assert len(profiles) == 1
        emb = profiles[0].embedding
        assert emb.dtype == np.float32
        assert emb.shape == (256,)
        assert np.all(np.isfinite(emb)), "Embedding should not contain NaN or Inf"
