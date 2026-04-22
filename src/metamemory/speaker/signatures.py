"""SQLite-backed voice signature store for speaker identification.

Stores speaker embeddings as BLOBs so that diarized speakers can be
matched against known voice signatures via cosine similarity. Uses WAL
mode for safe concurrent reads/writes from the recording pipeline.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import List, Optional

import numpy as np

from metamemory.speaker.models import SpeakerMatch, SpeakerProfile

logger = logging.getLogger(__name__)

# Default database location relative to the app data directory.
_DEFAULT_DB_DIR = Path("data")
_DEFAULT_DB_NAME = "speaker_signatures.db"


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors.

    Returns 0.0 when either vector has zero magnitude.
    """
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _embedding_to_blob(embedding: np.ndarray) -> bytes:
    """Serialize a float32 numpy array to bytes for SQLite BLOB storage."""
    return embedding.astype(np.float32).tobytes()


def _blob_to_embedding(blob: bytes) -> np.ndarray:
    """Deserialize a SQLite BLOB back to a float32 numpy array."""
    return np.frombuffer(blob, dtype=np.float32).copy()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS speaker_signatures (
    name        TEXT PRIMARY KEY,
    embedding   BLOB NOT NULL,
    num_samples INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class VoiceSignatureStore:
    """SQLite-backed store for named speaker voice embeddings.

    Thread-safe for concurrent reads via WAL mode. For concurrent writes
    from multiple threads, callers should use a single store instance or
    external serialization.

    Args:
        db_path: Path to the SQLite database file. Use ":memory:" for
                 transient testing.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._connect()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Open (or reopen) the database connection and ensure schema."""
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        conn.commit()
        self._conn = conn
        is_memory = self._db_path == ":memory:"
        logger.info(
            "Voice signature store opened (path=%s, memory=%s)",
            self._db_path,
            is_memory,
        )

    @property
    def conn(self) -> sqlite3.Connection:
        assert self._conn is not None, "Store is closed"
        return self._conn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.debug("Voice signature store closed")

    def save_signature(
        self,
        name: str,
        embedding: np.ndarray,
        averaged_from_segments: int = 1,
    ) -> None:
        """Persist a speaker voice signature.

        Args:
            name: Speaker name (unique key).
            embedding: Float32 embedding vector.
            averaged_from_segments: Number of audio segments that
                contributed to this embedding.
        """
        blob = _embedding_to_blob(embedding)
        self.conn.execute(
            """
            INSERT INTO speaker_signatures (name, embedding, num_samples)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                embedding   = excluded.embedding,
                num_samples = excluded.num_samples,
                updated_at  = datetime('now')
            """,
            (name, blob, averaged_from_segments),
        )
        self.conn.commit()
        logger.debug(
            "Saved signature for '%s' (segments=%d, dim=%d)",
            name,
            averaged_from_segments,
            len(embedding),
        )

    def load_signatures(self) -> List[SpeakerProfile]:
        """Return all stored speaker profiles."""
        rows = self.conn.execute(
            "SELECT name, embedding, num_samples FROM speaker_signatures "
            "ORDER BY name"
        ).fetchall()
        profiles: List[SpeakerProfile] = []
        for row in rows:
            profiles.append(
                SpeakerProfile(
                    name=row["name"],
                    embedding=_blob_to_embedding(row["embedding"]),
                    num_samples=row["num_samples"],
                )
            )
        logger.debug("Loaded %d signature(s)", len(profiles))
        return profiles

    def find_match(
        self,
        embedding: np.ndarray,
        threshold: float = 0.6,
    ) -> Optional[SpeakerMatch]:
        """Find the best-matching known speaker for an embedding.

        Computes cosine similarity against all stored profiles and returns
        the best match if it exceeds *threshold*.

        Args:
            embedding: Query embedding vector.
            threshold: Minimum cosine similarity for a match (0.0–1.0).

        Returns:
            A ``SpeakerMatch`` if a profile exceeds the threshold, else
            ``None``.
        """
        profiles = self.load_signatures()
        if not profiles:
            return None

        best_name: Optional[str] = None
        best_score: float = 0.0

        for profile in profiles:
            score = _cosine_similarity(embedding, profile.embedding)
            if score > best_score:
                best_score = score
                best_name = profile.name

        if best_score < threshold or best_name is None:
            logger.debug(
                "No match above threshold %.2f (best=%.4f)", threshold, best_score
            )
            return None

        # Classify confidence
        if best_score >= 0.85:
            confidence = "high"
        elif best_score >= 0.7:
            confidence = "medium"
        else:
            confidence = "low"

        logger.debug(
            "Match: '%s' score=%.4f confidence=%s", best_name, best_score, confidence
        )
        return SpeakerMatch(name=best_name, score=best_score, confidence=confidence)

    def delete_signature(self, name: str) -> bool:
        """Delete a speaker signature by name.

        Returns:
            True if a row was deleted, False if the name was not found.
        """
        cursor = self.conn.execute(
            "DELETE FROM speaker_signatures WHERE name = ?", (name,)
        )
        self.conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.debug("Deleted signature for '%s'", name)
        else:
            logger.debug("No signature found to delete for '%s'", name)
        return deleted

    def update_signature(self, name: str, embedding: np.ndarray) -> bool:
        """Update an existing speaker's embedding by running average.

        Performs incremental averaging: the stored embedding becomes the
        weighted average of the existing embedding and the new one,
        weighted by their respective sample counts.

        Args:
            name: Speaker name to update.
            embedding: New embedding to incorporate.

        Returns:
            True if an existing profile was updated, False if the name
            was not found (callers should use ``save_signature`` first).
        """
        row = self.conn.execute(
            "SELECT embedding, num_samples FROM speaker_signatures WHERE name = ?",
            (name,),
        ).fetchone()

        if row is None:
            logger.debug("No existing signature for '%s' to update", name)
            return False

        old_embedding = _blob_to_embedding(row["embedding"])
        old_count = row["num_samples"]
        new_count = old_count + 1

        # Weighted incremental average
        averaged = (old_embedding * old_count + embedding) / new_count

        blob = _embedding_to_blob(averaged)
        self.conn.execute(
            """
            UPDATE speaker_signatures
            SET embedding = ?, num_samples = ?, updated_at = datetime('now')
            WHERE name = ?
            """,
            (blob, new_count, name),
        )
        self.conn.commit()
        logger.debug(
            "Updated signature for '%s' (samples %d -> %d)",
            name,
            old_count,
            new_count,
        )
        return True

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __enter__(self) -> "VoiceSignatureStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __len__(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS cnt FROM speaker_signatures"
        ).fetchone()
        return row["cnt"]  # type: ignore[index]

    def __repr__(self) -> str:
        return f"VoiceSignatureStore(db_path={self._db_path!r})"
