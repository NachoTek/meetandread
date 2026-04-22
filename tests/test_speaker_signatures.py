"""Tests for the voice signature SQLite store."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from metamemory.speaker.models import SpeakerMatch, SpeakerProfile
from metamemory.speaker.signatures import VoiceSignatureStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_embedding(dim: int = 256, seed: int = 0) -> np.ndarray:
    """Return a deterministic random unit-norm embedding."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def store() -> VoiceSignatureStore:
    """In-memory store, auto-closed after each test."""
    with VoiceSignatureStore(":memory:") as s:
        yield s


@pytest.fixture
def persistent_store(tmp_path: Path) -> VoiceSignatureStore:
    """File-backed store in a temp directory."""
    db = tmp_path / "test_signatures.db"
    with VoiceSignatureStore(str(db)) as s:
        yield s


# ---------------------------------------------------------------------------
# Schema & connection
# ---------------------------------------------------------------------------

class TestStoreLifecycle:
    def test_in_memory_store_opens(self, store: VoiceSignatureStore) -> None:
        assert len(store) == 0

    def test_persistent_store_creates_file(self, persistent_store: VoiceSignatureStore) -> None:
        # Database file should exist (SQLite creates on connect)
        assert Path(persistent_store._db_path).exists()

    def test_close_and_reopen(self, tmp_path: Path) -> None:
        db = tmp_path / "lifecycle.db"
        emb = _random_embedding(seed=42)
        with VoiceSignatureStore(str(db)) as s:
            s.save_signature("Alice", emb)

        # Reopen — data should survive
        with VoiceSignatureStore(str(db)) as s:
            profiles = s.load_signatures()
            assert len(profiles) == 1
            assert profiles[0].name == "Alice"

    def test_context_manager(self) -> None:
        with VoiceSignatureStore(":memory:") as s:
            s.save_signature("Bob", _random_embedding(seed=1))
            assert len(s) == 1
        # After exit, connection is closed
        assert s._conn is None

    def test_repr(self, store: VoiceSignatureStore) -> None:
        r = repr(store)
        assert "VoiceSignatureStore" in r
        assert ":memory:" in r


# ---------------------------------------------------------------------------
# save_signature / load_signatures
# ---------------------------------------------------------------------------

class TestSaveAndLoad:
    def test_save_and_load_one(self, store: VoiceSignatureStore) -> None:
        emb = _random_embedding(seed=10)
        store.save_signature("Alice", emb, averaged_from_segments=3)

        profiles = store.load_signatures()
        assert len(profiles) == 1
        p = profiles[0]
        assert p.name == "Alice"
        assert p.num_samples == 3
        np.testing.assert_allclose(p.embedding, emb, atol=1e-6)

    def test_save_multiple(self, store: VoiceSignatureStore) -> None:
        for i, name in enumerate(["Alice", "Bob", "Carol"]):
            store.save_signature(name, _random_embedding(seed=i))

        profiles = store.load_signatures()
        assert len(profiles) == 3
        names = {p.name for p in profiles}
        assert names == {"Alice", "Bob", "Carol"}

    def test_upsert_overwrites(self, store: VoiceSignatureStore) -> None:
        emb_v1 = _random_embedding(seed=1)
        emb_v2 = _random_embedding(seed=2)
        store.save_signature("Alice", emb_v1)
        store.save_signature("Alice", emb_v2, averaged_from_segments=5)

        profiles = store.load_signatures()
        assert len(profiles) == 1
        np.testing.assert_allclose(profiles[0].embedding, emb_v2, atol=1e-6)
        assert profiles[0].num_samples == 5

    def test_load_empty(self, store: VoiceSignatureStore) -> None:
        assert store.load_signatures() == []


# ---------------------------------------------------------------------------
# find_match
# ---------------------------------------------------------------------------

class TestFindMatch:
    def test_exact_match(self, store: VoiceSignatureStore) -> None:
        emb = _random_embedding(seed=20)
        store.save_signature("Alice", emb)

        # Query with identical embedding → cosine ~1.0
        match = store.find_match(emb)
        assert match is not None
        assert match.name == "Alice"
        assert match.score > 0.99
        assert match.confidence == "high"

    def test_no_match_below_threshold(self, store: VoiceSignatureStore) -> None:
        store.save_signature("Alice", _random_embedding(seed=1))
        # Orthogonal embedding → low similarity
        query = _random_embedding(seed=999)
        result = store.find_match(query, threshold=0.9)
        assert result is None

    def test_best_match_picked(self, store: VoiceSignatureStore) -> None:
        alice_emb = _random_embedding(seed=10)
        bob_emb = _random_embedding(seed=20)
        store.save_signature("Alice", alice_emb)
        store.save_signature("Bob", bob_emb)

        # Query closer to Alice (seed 10 + tiny noise)
        rng = np.random.default_rng(10)
        query = alice_emb + rng.standard_normal(alice_emb.shape).astype(np.float32) * 0.01
        query = query / np.linalg.norm(query)

        match = store.find_match(query, threshold=0.5)
        assert match is not None
        assert match.name == "Alice"

    def test_empty_store_returns_none(self, store: VoiceSignatureStore) -> None:
        assert store.find_match(_random_embedding()) is None

    def test_confidence_levels(self, store: VoiceSignatureStore) -> None:
        emb = _random_embedding(seed=5)
        store.save_signature("Target", emb)

        # Slightly noisy copy → still high
        rng = np.random.default_rng(5)
        noisy = emb + rng.standard_normal(emb.shape).astype(np.float32) * 0.05
        noisy = noisy / np.linalg.norm(noisy)
        match = store.find_match(noisy, threshold=0.3)
        assert match is not None
        assert match.confidence in ("high", "medium", "low")


# ---------------------------------------------------------------------------
# delete_signature
# ---------------------------------------------------------------------------

class TestDeleteSignature:
    def test_delete_existing(self, store: VoiceSignatureStore) -> None:
        store.save_signature("Alice", _random_embedding(seed=1))
        assert len(store) == 1

        deleted = store.delete_signature("Alice")
        assert deleted is True
        assert len(store) == 0

    def test_delete_nonexistent(self, store: VoiceSignatureStore) -> None:
        deleted = store.delete_signature("Nobody")
        assert deleted is False


# ---------------------------------------------------------------------------
# update_signature (incremental averaging)
# ---------------------------------------------------------------------------

class TestUpdateSignature:
    def test_update_increments_sample_count(self, store: VoiceSignatureStore) -> None:
        emb = _random_embedding(seed=1)
        store.save_signature("Alice", emb, averaged_from_segments=2)

        new_emb = _random_embedding(seed=2)
        updated = store.update_signature("Alice", new_emb)
        assert updated is True

        profiles = store.load_signatures()
        assert len(profiles) == 1
        assert profiles[0].num_samples == 3  # 2 + 1

    def test_update_averages_embeddings(self, store: VoiceSignatureStore) -> None:
        emb1 = np.ones(4, dtype=np.float32)
        emb2 = np.ones(4, dtype=np.float32) * 3.0

        store.save_signature("Test", emb1, averaged_from_segments=1)
        store.update_signature("Test", emb2)

        profiles = store.load_signatures()
        expected = (emb1 * 1 + emb2) / 2  # weighted avg: (1*1 + 3) / 2 = 2.0
        np.testing.assert_allclose(profiles[0].embedding, expected, atol=1e-5)

    def test_update_nonexistent_returns_false(self, store: VoiceSignatureStore) -> None:
        result = store.update_signature("Nobody", _random_embedding())
        assert result is False

    def test_update_preserves_other_speakers(self, store: VoiceSignatureStore) -> None:
        store.save_signature("Alice", _random_embedding(seed=1))
        store.save_signature("Bob", _random_embedding(seed=2))

        store.update_signature("Alice", _random_embedding(seed=3))
        profiles = store.load_signatures()
        assert len(profiles) == 2
        names = {p.name for p in profiles}
        assert names == {"Alice", "Bob"}


# ---------------------------------------------------------------------------
# WAL mode verification
# ---------------------------------------------------------------------------

class TestWalMode:
    def test_wal_mode_enabled(self, persistent_store: VoiceSignatureStore) -> None:
        row = persistent_store.conn.execute("PRAGMA journal_mode").fetchone()
        # SQLite returns the mode as a string column
        mode = list(row.values())[0] if hasattr(row, "values") else row[0]
        assert str(mode).lower() == "wal"
