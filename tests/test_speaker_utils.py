"""Tests for the shared speaker utility functions."""

from __future__ import annotations

import numpy as np
import pytest

from meetandread.speaker.utils import cosine_similarity


# ---------------------------------------------------------------------------
# cosine_similarity — basic cases
# ---------------------------------------------------------------------------


class TestCosineSimilarityBasic:
    def test_identical_vectors(self) -> None:
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    def test_identical_unit_vectors(self) -> None:
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_general_angle(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 1.0], dtype=np.float32)
        expected = 1.0 / np.sqrt(2.0)
        assert cosine_similarity(a, b) == pytest.approx(expected)

    def test_high_dimensional(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.standard_normal(512).astype(np.float32)
        b = a + rng.standard_normal(512).astype(np.float32) * 0.01
        assert cosine_similarity(a, b) > 0.99


# ---------------------------------------------------------------------------
# cosine_similarity — zero vectors
# ---------------------------------------------------------------------------


class TestCosineSimilarityZeroVectors:
    def test_zero_a(self) -> None:
        a = np.zeros(3, dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(a, b) == 0.0

    def test_zero_b(self) -> None:
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.zeros(3, dtype=np.float32)
        assert cosine_similarity(a, b) == 0.0

    def test_both_zero(self) -> None:
        a = np.zeros(3, dtype=np.float32)
        b = np.zeros(3, dtype=np.float32)
        assert cosine_similarity(a, b) == 0.0


# ---------------------------------------------------------------------------
# cosine_similarity — NaN / inf inputs
# ---------------------------------------------------------------------------


class TestCosineSimilarityNonFinite:
    def test_nan_in_a(self) -> None:
        a = np.array([1.0, float("nan"), 3.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(a, b) == 0.0

    def test_nan_in_b(self) -> None:
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([1.0, float("nan"), 3.0], dtype=np.float32)
        assert cosine_similarity(a, b) == 0.0

    def test_inf_in_a(self) -> None:
        a = np.array([1.0, float("inf"), 3.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(a, b) == 0.0

    def test_neg_inf_in_b(self) -> None:
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([1.0, float("-inf"), 3.0], dtype=np.float32)
        assert cosine_similarity(a, b) == 0.0

    def test_all_nan(self) -> None:
        a = np.full(3, float("nan"), dtype=np.float32)
        b = np.full(3, float("nan"), dtype=np.float32)
        assert cosine_similarity(a, b) == 0.0


# ---------------------------------------------------------------------------
# cosine_similarity — shape validation
# ---------------------------------------------------------------------------


class TestCosineSimilarityShapeValidation:
    def test_2d_array_raises(self) -> None:
        a = np.array([[1.0, 2.0]], dtype=np.float32)
        b = np.array([1.0, 2.0], dtype=np.float32)
        with pytest.raises(ValueError, match="1-D"):
            cosine_similarity(a, b)

    def test_2d_array_b_raises(self) -> None:
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([[1.0, 2.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="1-D"):
            cosine_similarity(a, b)

    def test_length_mismatch_raises(self) -> None:
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([1.0, 2.0], dtype=np.float32)
        with pytest.raises(ValueError, match="mismatch"):
            cosine_similarity(a, b)

    def test_scalar_raises(self) -> None:
        a = np.float32(1.0)
        b = np.float32(1.0)
        with pytest.raises(ValueError, match="1-D"):
            cosine_similarity(a, b)


# ---------------------------------------------------------------------------
# cosine_similarity — integration with VoiceSignatureStore
# ---------------------------------------------------------------------------


class TestCosineSimilarityIntegration:
    """Verify the shared helper produces the same results as callers expect."""

    def test_store_find_match_uses_shared_similarity(self) -> None:
        """End-to-end: VoiceSignatureStore.find_match still works."""
        from meetandread.speaker.signatures import VoiceSignatureStore

        with VoiceSignatureStore(":memory:") as store:
            emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            store.save_signature("Alice", emb)

            # Exact match
            result = store.find_match(emb, threshold=0.5)
            assert result is not None
            assert result.name == "Alice"
            assert result.score == pytest.approx(1.0)

            # No match below threshold
            ortho = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            assert store.find_match(ortho, threshold=0.5) is None
