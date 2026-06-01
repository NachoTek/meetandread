"""Shared speaker utility functions.

Centralizes helpers used across the speaker identification pipeline so
that edge-case handling (zero vectors, NaN/inf inputs, shape mismatches)
is tested and maintained in one place.
"""

from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D float vectors.

    Returns ``0.0`` for:
    * zero-magnitude vectors (no directional information)
    * vectors containing NaN or inf values
    * non-finite computed results

    Raises:
        ValueError: If either input is not 1-D or the vectors have
            different lengths.
    """
    # Shape validation — must be 1-D
    if a.ndim != 1:
        raise ValueError(f"Expected 1-D array for 'a', got {a.ndim}-D")
    if b.ndim != 1:
        raise ValueError(f"Expected 1-D array for 'b', got {b.ndim}-D")
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"Vector length mismatch: a has {a.shape[0]}, b has {b.shape[0]}"
        )

    # Fast path for non-finite inputs — return safe no-match rather than
    # propagating NaN through the dot product / norm computation.
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return 0.0

    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    result = dot / (norm_a * norm_b)

    # Guard against pathological floating-point results
    if not np.isfinite(result):
        return 0.0

    return result
