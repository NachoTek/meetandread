"""Speaker re-identification accuracy benchmark using synthetic embeddings.

Validates VoiceSignatureStore.find_match() achieves >= 90% accuracy across
multiple scenarios: exact match, noisy match, no-match rejection, threshold
sensitivity, and multi-speaker discrimination. Uses deterministic random
embeddings (no real audio or model inference required).

Results are written to src/meetandread/performance/test_data/speaker_reid_results.txt.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from meetandread.speaker.signatures import VoiceSignatureStore
from meetandread.speaker.utils import cosine_similarity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RESULTS_PATH = Path("src/meetandread/performance/test_data/speaker_reid_results.txt")
EMBEDDING_DIM = 256
DEFAULT_THRESHOLD = 0.6


def _random_embedding(dim: int = EMBEDDING_DIM, seed: int = 0) -> np.ndarray:
    """Return a deterministic random unit-norm embedding."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _noisy_copy(embedding: np.ndarray, noise_std: float = 0.05, seed: int = 0) -> np.ndarray:
    """Return a perturbed copy of an embedding, re-normalised to unit length."""
    rng = np.random.default_rng(seed)
    noisy = embedding + rng.standard_normal(embedding.shape).astype(np.float32) * noise_std
    return noisy / np.linalg.norm(noisy)


def _embedding_at_similarity(
    target: np.ndarray, similarity: float, dim: int = EMBEDDING_DIM, seed: int = 0
) -> np.ndarray:
    """Return a unit-norm embedding at exactly *similarity* cosine distance from *target*.

    Constructed by blending the target with an orthogonal component so that
    cos(query, target) == similarity.
    """
    if similarity >= 1.0:
        return target.copy()
    if similarity <= 0.0:
        # Fully orthogonal — return a random unit vector projected orthogonal
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(dim).astype(np.float32)
        v = v - np.dot(v, target) * target
        norm = np.linalg.norm(v)
        if norm < 1e-8:
            return _random_embedding(dim, seed)
        return v / norm

    # query = similarity * target + sqrt(1 - similarity^2) * orth
    rng = np.random.default_rng(seed)
    orth = rng.standard_normal(dim).astype(np.float32)
    orth = orth - np.dot(orth, target) * target  # Gram-Schmidt
    norm = np.linalg.norm(orth)
    if norm < 1e-8:
        orth = _random_embedding(dim, seed)
        orth = orth - np.dot(orth, target) * target
        norm = np.linalg.norm(orth)
    orth = orth / norm
    query = similarity * target + np.sqrt(1.0 - similarity ** 2) * orth
    return query / np.linalg.norm(query)


def _orthogonal_embedding(
    references: list[np.ndarray], dim: int = EMBEDDING_DIM, seed: int = 0
) -> np.ndarray:
    """Return a unit-norm embedding approximately orthogonal to all *references*."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    for ref in references:
        v = v - np.dot(v, ref) * ref
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        # Degenerate — just use a different seed
        return _random_embedding(dim, seed + 10000)
    return v / norm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store() -> VoiceSignatureStore:
    """In-memory store, auto-closed after each test."""
    with VoiceSignatureStore(":memory:") as s:
        yield s


# ---------------------------------------------------------------------------
# Scenario 1: Exact match
# ---------------------------------------------------------------------------

class TestExactMatch:
    """Save embedding with seed N, query with identical embedding."""

    def test_exact_match_returns_high_score(self, store: VoiceSignatureStore) -> None:
        emb = _random_embedding(seed=42)
        store.save_signature("Speaker_A", emb)

        match = store.find_match(emb)
        assert match is not None, "Exact match should be found"
        assert match.name == "Speaker_A"
        assert match.score > 0.99, f"Exact match score should be ~1.0, got {match.score}"
        assert match.confidence == "high"

    @pytest.mark.parametrize("seed", [0, 7, 42, 99, 255])
    def test_exact_match_multiple_seeds(
        self, store: VoiceSignatureStore, seed: int
    ) -> None:
        name = f"speaker_{seed}"
        emb = _random_embedding(seed=seed)
        store.save_signature(name, emb)

        match = store.find_match(emb)
        assert match is not None
        assert match.name == name
        assert match.score > 0.999


# ---------------------------------------------------------------------------
# Scenario 2: Noisy match
# ---------------------------------------------------------------------------

class TestNoisyMatch:
    """Query with a slightly perturbed copy of a stored embedding."""

    @pytest.mark.parametrize("noise_std", [0.01, 0.03, 0.05])
    def test_noisy_match_found(
        self, store: VoiceSignatureStore, noise_std: float
    ) -> None:
        emb = _random_embedding(seed=10)
        store.save_signature("NoisySpeaker", emb)

        query = _noisy_copy(emb, noise_std=noise_std, seed=100)
        match = store.find_match(query, threshold=DEFAULT_THRESHOLD)
        assert match is not None, (
            f"Noisy match (std={noise_std}) should be found above threshold "
            f"{DEFAULT_THRESHOLD}"
        )
        assert match.name == "NoisySpeaker"

    def test_high_noise_below_threshold(self, store: VoiceSignatureStore) -> None:
        """With very high noise (std=0.10), similarity may drop below 0.6."""
        emb = _random_embedding(seed=10)
        store.save_signature("NoisySpeaker", emb)

        query = _noisy_copy(emb, noise_std=0.10, seed=100)
        sim = cosine_similarity(emb, query)
        # At noise_std=0.10 in 256-dim, similarity typically drops to ~0.4-0.6
        # This is expected — the test documents the boundary behavior.
        assert 0.3 < sim < 0.8, f"Expected sim in (0.3, 0.8) at std=0.10, got {sim}"

    def test_noisy_match_similarity_range(self, store: VoiceSignatureStore) -> None:
        """Verify that moderate noise keeps similarity above threshold."""
        emb = _random_embedding(seed=20)
        store.save_signature("Speaker", emb)

        query = _noisy_copy(emb, noise_std=0.05, seed=200)
        sim = cosine_similarity(emb, query)
        # In 256-dim, std=0.05 noise typically yields sim in [0.7, 0.9]
        assert sim > 0.6, f"Noisy copy similarity should be >0.6, got {sim}"


# ---------------------------------------------------------------------------
# Scenario 3: No match
# ---------------------------------------------------------------------------

class TestNoMatch:
    """Query with orthogonal embedding → no match expected."""

    def test_orthogonal_no_match(self, store: VoiceSignatureStore) -> None:
        emb_a = _random_embedding(seed=1)
        emb_b = _random_embedding(seed=2)
        store.save_signature("A", emb_a)
        store.save_signature("B", emb_b)

        query = _orthogonal_embedding([emb_a, emb_b], seed=500)
        match = store.find_match(query, threshold=DEFAULT_THRESHOLD)
        # Orthogonal embeddings should have low cosine similarity
        sim_a = cosine_similarity(query, emb_a)
        sim_b = cosine_similarity(query, emb_b)
        max_sim = max(sim_a, sim_b)
        if max_sim < DEFAULT_THRESHOLD:
            assert match is None, (
                f"Orthogonal query should not match (max_sim={max_sim:.4f})"
            )

    def test_random_unrelated_no_match(self, store: VoiceSignatureStore) -> None:
        """High-seed random embeddings are unlikely to match stored ones."""
        for i in range(5):
            store.save_signature(f"spk_{i}", _random_embedding(seed=i * 10))

        # Very different seed → statistically independent embedding
        query = _random_embedding(seed=99999)
        match = store.find_match(query, threshold=0.9)
        assert match is None, "Unrelated embedding should not match at high threshold"


# ---------------------------------------------------------------------------
# Scenario 4: Threshold sensitivity sweep
# ---------------------------------------------------------------------------

class TestThresholdSweep:
    """Verify find_match behavior at various cosine similarity levels."""

    @pytest.mark.parametrize("similarity", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9])
    def test_threshold_sweep(
        self, store: VoiceSignatureStore, similarity: float
    ) -> None:
        emb = _random_embedding(seed=77)
        store.save_signature("Target", emb)

        query = _embedding_at_similarity(emb, similarity, seed=777)

        # Verify the constructed similarity is close to requested
        actual_sim = cosine_similarity(emb, query)
        assert abs(actual_sim - similarity) < 0.02, (
            f"Requested sim={similarity}, got {actual_sim:.4f}"
        )

        match = store.find_match(query, threshold=DEFAULT_THRESHOLD)
        if similarity >= DEFAULT_THRESHOLD:
            assert match is not None, (
                f"Should match at sim={similarity} (threshold={DEFAULT_THRESHOLD})"
            )
            assert match.name == "Target"
        else:
            assert match is None, (
                f"Should NOT match at sim={similarity} (threshold={DEFAULT_THRESHOLD})"
            )


# ---------------------------------------------------------------------------
# Scenario 5: Multi-speaker discrimination
# ---------------------------------------------------------------------------

class TestMultiSpeakerDiscrimination:
    """Store 5 speakers with distinct embeddings, query each one."""

    NUM_SPEAKERS = 5

    @pytest.fixture
    def multi_store(self) -> VoiceSignatureStore:
        """Store pre-loaded with 5 distinct speakers."""
        with VoiceSignatureStore(":memory:") as s:
            for i in range(self.NUM_SPEAKERS):
                s.save_signature(f"speaker_{i}", _random_embedding(seed=i * 100))
            yield s

    def test_all_speakers_correctly_identified(
        self, multi_store: VoiceSignatureStore
    ) -> None:
        correct = 0
        for i in range(self.NUM_SPEAKERS):
            query = _random_embedding(seed=i * 100)
            match = multi_store.find_match(query)
            if match is not None and match.name == f"speaker_{i}":
                correct += 1
        assert correct == self.NUM_SPEAKERS, (
            f"Expected all {self.NUM_SPEAKERS} speakers identified, got {correct}"
        )

    def test_noisy_queries_still_correct(
        self, multi_store: VoiceSignatureStore
    ) -> None:
        correct = 0
        for i in range(self.NUM_SPEAKERS):
            query = _noisy_copy(
                _random_embedding(seed=i * 100), noise_std=0.05, seed=i + 1000
            )
            match = multi_store.find_match(query)
            if match is not None and match.name == f"speaker_{i}":
                correct += 1
        assert correct == self.NUM_SPEAKERS, (
            f"Expected all {self.NUM_SPEAKERS} noisy queries correct, got {correct}"
        )


# ---------------------------------------------------------------------------
# Accuracy trial: 100 trials → assert ≥ 90%
# ---------------------------------------------------------------------------

class TestAccuracyTrials:
    """Run 100 trials of match / no-match to compute accuracy."""

    def test_overall_accuracy_ge_90(self, store: VoiceSignatureStore) -> None:
        """100 trials: store speakers, query exact/noisy/orthogonal → ≥ 90% accuracy."""
        results: list[dict] = []
        num_trials = 100

        for trial in range(num_trials):
            # Register 3 speakers
            seeds = [trial * 3, trial * 3 + 1, trial * 3 + 2]
            for s in seeds:
                store.save_signature(f"spk_{s}", _random_embedding(seed=s))

            # --- Match trial: query the first speaker exactly ---
            query_exact = _random_embedding(seed=seeds[0])
            match = store.find_match(query_exact, threshold=DEFAULT_THRESHOLD)
            results.append({
                "trial": trial,
                "type": "exact_match",
                "expected": f"spk_{seeds[0]}",
                "got": match.name if match else None,
                "correct": match is not None and match.name == f"spk_{seeds[0]}",
            })

            # --- Noisy match trial ---
            query_noisy = _noisy_copy(
                _random_embedding(seed=seeds[1]), noise_std=0.05, seed=trial + 5000
            )
            match = store.find_match(query_noisy, threshold=DEFAULT_THRESHOLD)
            results.append({
                "trial": trial,
                "type": "noisy_match",
                "expected": f"spk_{seeds[1]}",
                "got": match.name if match else None,
                "correct": match is not None and match.name == f"spk_{seeds[1]}",
            })

            # --- No-match trial: orthogonal query ---
            refs = [_random_embedding(seed=s) for s in seeds]
            query_orth = _orthogonal_embedding(refs, seed=trial + 9000)
            match = store.find_match(query_orth, threshold=DEFAULT_THRESHOLD)
            sims = [cosine_similarity(query_orth, r) for r in refs]
            max_sim = max(sims)
            expected_none = max_sim < DEFAULT_THRESHOLD
            results.append({
                "trial": trial,
                "type": "no_match",
                "expected": None,
                "got": match.name if match else None,
                "correct": expected_none and match is None,
            })

            # Clean up speakers for next trial (fresh store state)
            for s in seeds:
                store.delete_signature(f"spk_{s}")

        total = len(results)
        correct = sum(1 for r in results if r["correct"])
        accuracy = correct / total

        # Write results to file
        _write_results(results, total, correct, accuracy)

        assert accuracy >= 0.90, (
            f"Speaker re-id accuracy {accuracy:.2%} ({correct}/{total}) "
            f"is below the 90% target"
        )


# ---------------------------------------------------------------------------
# Results writer
# ---------------------------------------------------------------------------

def _write_results(
    results: list[dict], total: int, correct: int, accuracy: float
) -> None:
    """Write benchmark results to the results file."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("SPEAKER RE-IDENTIFICATION BENCHMARK RESULTS")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"Embedding dimension: {EMBEDDING_DIM}")
    lines.append(f"Default threshold:   {DEFAULT_THRESHOLD}")
    lines.append(f"Total trials:        {total}")
    lines.append(f"Correct:             {correct}")
    lines.append(f"Accuracy:            {accuracy:.2%}")
    lines.append(f"Target:              ≥ 90%")
    target_met = accuracy >= 0.90
    lines.append(f"Target met:          {'YES [✓]' if target_met else 'NO [X]'}")
    lines.append("")

    # Per-scenario breakdown
    scenarios: dict[str, dict] = {}
    for r in results:
        t = r["type"]
        if t not in scenarios:
            scenarios[t] = {"total": 0, "correct": 0}
        scenarios[t]["total"] += 1
        if r["correct"]:
            scenarios[t]["correct"] += 1

    lines.append("PER-SCENARIO BREAKDOWN")
    lines.append("-" * 40)
    for scenario, data in sorted(scenarios.items()):
        acc = data["correct"] / data["total"] if data["total"] else 0
        lines.append(f"  {scenario:20s}  {data['correct']:3d}/{data['total']:3d}  ({acc:.1%})")
    lines.append("")

    # Sample failures (up to 10)
    failures = [r for r in results if not r["correct"]]
    if failures:
        lines.append("SAMPLE FAILURES (up to 10)")
        lines.append("-" * 40)
        for f in failures[:10]:
            lines.append(
                f"  trial={f['trial']:3d}  type={f['type']:12s}  "
                f"expected={f['expected']}  got={f['got']}"
            )
    else:
        lines.append("NO FAILURES — all trials correct.")
    lines.append("")

    # Threshold sensitivity summary
    lines.append("THRESHOLD SENSITIVITY (from sweep tests)")
    lines.append("-" * 40)
    for sim_level in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
        expected_match = sim_level >= DEFAULT_THRESHOLD
        lines.append(
            f"  similarity={sim_level:.1f}  threshold={DEFAULT_THRESHOLD}  "
            f"expected={'MATCH' if expected_match else 'NO MATCH'}"
        )
    lines.append("")

    if not target_met:
        lines.append("=" * 70)
        lines.append("GAP ANALYSIS — Accuracy below target")
        lines.append("=" * 70)
        lines.append(f"  Actual accuracy:   {accuracy:.2%}")
        lines.append(f"  Target accuracy:   90%")
        lines.append(f"  Gap:               {(1.0 - accuracy) * 100:.2f}% above target")
        lines.append("")
        lines.append("LIKELY CAUSES:")
        lines.append("  1. Cosine similarity threshold may need tuning for the")
        lines.append("     specific embedding dimension and noise profile.")
        lines.append("  2. Orthogonal embedding construction may not always achieve")
        lines.append("     true orthogonality in 256-dim space with small seed space.")
        lines.append("  3. Noisy match with std=0.05 may push some queries below")
        lines.append("     the default threshold of 0.6.")
        lines.append("")

    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
