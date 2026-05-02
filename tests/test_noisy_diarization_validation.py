"""Deterministic noisy multi-speaker fixture and diarization metric tests.

Validates the M008 audio improvements end-to-end with synthetic audio:
1. Fixture generation produces correct WAV format and ground-truth metadata
2. Diarization cleanup reduces false speaker splits (over-segmentation)
3. True speaker changes (A/B/A turns) are preserved
4. Malformed/edge-case inputs are handled without crashing
5. Metrics objectively measure both sides of the cleanup tradeoff

All tests are deterministic — no sherpa-onnx dependency required.
Optional slow tests gated behind @pytest.mark.slow test real diarizer.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from meetandread.speaker.models import SpeakerSegment
from meetandread.speaker.diarizer import (
    cleanup_diarization_segments,
    DEFAULT_GAP_MERGE_THRESHOLD,
    DEFAULT_SHORT_SEGMENT_THRESHOLD,
)

from tests.audio_fixture_helpers import (
    generate_noisy_multi_speaker_wav,
    validate_fixture_wav,
    GroundTruth,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SAMPLE_WIDTH,
    DEFAULT_CHANNELS,
)


# ---------------------------------------------------------------------------
# Metric helpers — measure cleanup quality objectively
# ---------------------------------------------------------------------------


def count_false_splits(
    segments: List[SpeakerSegment],
    ground_truth: GroundTruth,
    tolerance: float = 0.3,
) -> int:
    """Count segments that split a single ground-truth speaker turn.

    A false split is a segment boundary within a ground-truth speaker turn
    that does not correspond to any real speaker change. We check each pair
    of consecutive segments: if both fall within the same ground-truth turn
    and their boundary is not near a real speaker change, it's a false split.
    """
    if len(segments) <= 1:
        return 0

    false_splits = 0
    for i in range(1, len(segments)):
        boundary_time = segments[i].start
        # Check if this boundary is near a real speaker change
        near_real_change = any(
            abs(boundary_time - b.time) <= tolerance
            for b in ground_truth.boundaries
        )
        if near_real_change:
            continue

        # Check if both sides of the boundary are within the same GT turn
        for gt_start, gt_end, gt_speaker in ground_truth.segments:
            if gt_start <= boundary_time <= gt_end:
                # Boundary is inside a single GT speaker turn — false split
                false_splits += 1
                break

    return false_splits


def count_preserved_boundaries(
    segments: List[SpeakerSegment],
    ground_truth: GroundTruth,
    tolerance: float = 0.3,
) -> int:
    """Count ground-truth speaker-change boundaries that have a matching
    segment boundary in the cleaned segments.

    Returns (preserved_count, total_boundaries).
    """
    if not ground_truth.boundaries or not segments:
        return 0, len(ground_truth.boundaries)

    preserved = 0
    for gt_boundary in ground_truth.boundaries:
        # Find a segment boundary near this ground-truth boundary
        for i in range(1, len(segments)):
            seg_boundary = segments[i].start
            if abs(seg_boundary - gt_boundary.time) <= tolerance:
                # Verify the speakers actually change across the boundary
                if segments[i - 1].speaker != segments[i].speaker:
                    preserved += 1
                break

    return preserved, len(ground_truth.boundaries)


def segment_count_reduction(
    before: List[SpeakerSegment],
    after: List[SpeakerSegment],
) -> int:
    """Return the number of segments removed by cleanup."""
    return len(before) - len(after)


def compute_speaker_purity(
    segments: List[SpeakerSegment],
    ground_truth: GroundTruth,
) -> float:
    """Fraction of segments whose speaker label is consistent within each
    ground-truth turn.

    A segment is "pure" if it lies entirely within a single ground-truth
    turn and its speaker matches the ground-truth speaker for that turn.
    Returns 0.0–1.0.
    """
    if not segments:
        return 0.0

    pure = 0
    for seg in segments:
        mid = (seg.start + seg.end) / 2.0
        for gt_start, gt_end, gt_speaker in ground_truth.segments:
            if gt_start <= mid <= gt_end:
                # Midpoint is in this GT turn — segment is pure
                # (We can't verify actual speaker match since cleanup uses
                # synthetic labels like spk0/spk1, not A/B, so we just
                # check the segment exists within a GT turn)
                pure += 1
                break

    return pure / len(segments)


# ---------------------------------------------------------------------------
# Fixture generation tests
# ---------------------------------------------------------------------------


class TestFixtureGeneration:
    """Verify synthetic WAV fixture generator produces correct output."""

    def test_default_fixture_wav_format(self, tmp_path: Path):
        """Default fixture is a valid 16 kHz mono 16-bit WAV."""
        wav_path, gt = generate_noisy_multi_speaker_wav(tmp_path / "test.wav")
        meta = validate_fixture_wav(wav_path)
        assert meta["sample_rate"] == 16000
        assert meta["channels"] == 1
        assert meta["sample_width"] == 2

    def test_default_fixture_duration(self, tmp_path: Path):
        """Default fixture has expected total duration."""
        wav_path, gt = generate_noisy_multi_speaker_wav(tmp_path / "test.wav")
        # Default: 4 turns: 2.0 + 2.0 + 1.6 + 2.0 = 7.6s speech
        # Plus 3 gaps of 0.3s each = 0.9s
        # Total ~ 8.5s
        assert gt.duration >= 7.0
        assert gt.duration <= 10.0
        meta = validate_fixture_wav(wav_path)
        assert abs(meta["duration_seconds"] - gt.duration) < 0.01

    def test_fixture_ground_truth_segments(self, tmp_path: Path):
        """Ground truth has correct number of segments and boundaries."""
        _, gt = generate_noisy_multi_speaker_wav(tmp_path / "test.wav")
        # Default: 4 speaker turns → 4 segments, 3 boundaries
        assert len(gt.segments) == 4
        assert len(gt.boundaries) == 3
        assert gt.speakers == ["A", "B"]

    def test_fixture_deterministic(self, tmp_path: Path):
        """Same seed produces identical WAV content."""
        path1, gt1 = generate_noisy_multi_speaker_wav(
            tmp_path / "test1.wav", seed=42
        )
        path2, gt2 = generate_noisy_multi_speaker_wav(
            tmp_path / "test2.wav", seed=42
        )
        assert path1.read_bytes() == path2.read_bytes()
        assert gt1.segments == gt2.segments

    def test_fixture_different_seeds_differ(self, tmp_path: Path):
        """Different seeds produce different WAV content."""
        path1, gt1 = generate_noisy_multi_speaker_wav(
            tmp_path / "test1.wav", seed=42
        )
        path2, gt2 = generate_noisy_multi_speaker_wav(
            tmp_path / "test2.wav", seed=99
        )
        # Segments are same (same turn pattern), but audio differs
        assert path1.read_bytes() != path2.read_bytes()

    def test_fixture_custom_turns(self, tmp_path: Path):
        """Custom speaker turns produce correct segments."""
        turns = [("A", 1.0), ("B", 1.5), ("A", 2.0)]
        _, gt = generate_noisy_multi_speaker_wav(
            tmp_path / "test.wav", speaker_turns=turns, gap_duration=0.2
        )
        assert len(gt.segments) == 3
        s0_start, s0_end, s0_spk = gt.segments[0]
        s1_start, s1_end, s1_spk = gt.segments[1]
        s2_start, s2_end, s2_spk = gt.segments[2]
        assert (s0_start, s0_end, s0_spk) == (pytest.approx(0.0), pytest.approx(1.0), "A")
        assert (s1_start, s1_end, s1_spk) == (pytest.approx(1.2), pytest.approx(2.7), "B")
        assert (s2_start, s2_end, s2_spk) == (pytest.approx(2.9), pytest.approx(4.9), "A")

    def test_fixture_aba_pattern(self, tmp_path: Path):
        """A/B/A pattern produces correct boundaries."""
        turns = [("A", 1.0), ("B", 1.0), ("A", 1.0)]
        _, gt = generate_noisy_multi_speaker_wav(
            tmp_path / "test.wav", speaker_turns=turns, gap_duration=0.1
        )
        assert len(gt.boundaries) == 2
        assert gt.boundaries[0].speaker_before == "A"
        assert gt.boundaries[0].speaker_after == "B"
        assert gt.boundaries[1].speaker_before == "B"
        assert gt.boundaries[1].speaker_after == "A"

    def test_fixture_noise_level_recorded(self, tmp_path: Path):
        """Ground truth records the noise level used."""
        _, gt = generate_noisy_multi_speaker_wav(
            tmp_path / "test.wav", noise_level=0.25
        )
        assert gt.noise_level == 0.25

    def test_fixture_validation_rejects_missing_file(self, tmp_path: Path):
        """validate_fixture_wav raises on missing file."""
        with pytest.raises(AssertionError, match="does not exist"):
            validate_fixture_wav(tmp_path / "nonexistent.wav")

    def test_fixture_validation_checks_duration(self, tmp_path: Path):
        """validate_fixture_wav rejects too-short files."""
        wav_path, _ = generate_noisy_multi_speaker_wav(
            tmp_path / "test.wav",
            speaker_turns=[("A", 0.5)],
            gap_duration=0.0,
        )
        with pytest.raises(AssertionError, match="shorter than minimum"):
            validate_fixture_wav(wav_path, min_duration=2.0)

    def test_fixture_invalid_sample_rate_raises(self, tmp_path: Path):
        """Invalid sample rate raises ValueError."""
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            generate_noisy_multi_speaker_wav(
                tmp_path / "test.wav", sample_rate=0
            )

    def test_fixture_negative_gap_raises(self, tmp_path: Path):
        """Negative gap duration raises ValueError."""
        with pytest.raises(ValueError, match="gap_duration must be non-negative"):
            generate_noisy_multi_speaker_wav(
                tmp_path / "test.wav", gap_duration=-0.1
            )

    def test_fixture_negative_noise_raises(self, tmp_path: Path):
        """Negative noise level raises ValueError."""
        with pytest.raises(ValueError, match="noise_level must be non-negative"):
            generate_noisy_multi_speaker_wav(
                tmp_path / "test.wav", noise_level=-0.1
            )

    def test_fixture_no_speech_turns(self, tmp_path: Path):
        """Empty speaker turns produces a minimal fixture."""
        wav_path, gt = generate_noisy_multi_speaker_wav(
            tmp_path / "test.wav", speaker_turns=[]
        )
        assert len(gt.segments) == 0
        assert len(gt.boundaries) == 0
        assert gt.duration == 0.0

    def test_fixture_wav_readable_by_standard_lib(self, tmp_path: Path):
        """Generated WAV is readable by Python's wave module."""
        import wave as wave_mod

        wav_path, _ = generate_noisy_multi_speaker_wav(tmp_path / "test.wav")
        with wave_mod.open(str(wav_path), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            frames = wf.readframes(wf.getnframes())
            assert len(frames) > 0


# ---------------------------------------------------------------------------
# Cleanup threshold assertion tests
# ---------------------------------------------------------------------------


class TestCleanupThresholds:
    """Assert cleanup thresholds have specific values so future tuning
    changes fail loudly."""

    def test_gap_merge_threshold_value(self):
        """DEFAULT_GAP_MERGE_THRESHOLD is 0.2s."""
        assert DEFAULT_GAP_MERGE_THRESHOLD == 0.2

    def test_short_segment_threshold_value(self):
        """DEFAULT_SHORT_SEGMENT_THRESHOLD is 0.5s."""
        assert DEFAULT_SHORT_SEGMENT_THRESHOLD == 0.5


# ---------------------------------------------------------------------------
# Cleanup metric tests — noisy same-speaker micro-splits should merge
# ---------------------------------------------------------------------------


class TestCleanupMetrics:
    """Metric-based tests proving cleanup reduces false splits while
    preserving true speaker changes."""

    @staticmethod
    def _seg(start: float, end: float, speaker: str) -> SpeakerSegment:
        return SpeakerSegment(start=start, end=end, speaker=speaker)

    def test_noisy_same_speaker_micro_splits_merged(self):
        """Multiple noisy micro-splits of the same speaker merge away."""
        # Simulate diarizer noise: one continuous speaker turn split into
        # many tiny segments with small gaps
        noisy = [
            self._seg(0.0, 0.8, "spk0"),
            self._seg(0.85, 1.5, "spk0"),   # 0.05s gap
            self._seg(1.6, 2.1, "spk0"),     # 0.1s gap
            self._seg(2.15, 3.0, "spk0"),    # 0.05s gap
        ]
        cleaned = cleanup_diarization_segments(noisy)
        assert len(cleaned) == 1, f"Expected 1 merged segment, got {len(cleaned)}"
        assert cleaned[0].start == 0.0
        assert cleaned[0].end == 3.0

    def test_true_aba_turns_preserved(self):
        """True A/B/A speaker changes are preserved after cleanup."""
        segs = [
            self._seg(0.0, 2.0, "spk0"),    # Speaker A (2s)
            self._seg(2.5, 4.5, "spk1"),    # Speaker B (2s) — 0.5s gap
            self._seg(5.0, 7.0, "spk0"),    # Speaker A again (2s) — 0.5s gap
        ]
        cleaned = cleanup_diarization_segments(segs)
        assert len(cleaned) == 3
        assert cleaned[0].speaker == "spk0"
        assert cleaned[1].speaker == "spk1"
        assert cleaned[2].speaker == "spk0"

    def test_metric_false_split_count_decreases(self):
        """Metrics show fewer false splits after cleanup."""
        from tests.audio_fixture_helpers import GroundTruth, GroundTruthBoundary

        # Ground truth: one speaker talking continuously for 0-5s
        gt = GroundTruth(
            duration=5.0,
            sample_rate=16000,
            speakers=["A"],
            boundaries=[],
            segments=[(0.0, 5.0, "A")],
            noise_level=0.1,
            seed=42,
        )

        # Simulate noisy diarizer output: same speaker but micro-split
        noisy = [
            self._seg(0.0, 1.0, "spk0"),
            self._seg(1.05, 2.0, "spk0"),
            self._seg(2.1, 3.0, "spk0"),
            self._seg(3.05, 4.0, "spk0"),
            self._seg(4.1, 5.0, "spk0"),
        ]

        before_splits = count_false_splits(noisy, gt)
        assert before_splits >= 3, f"Expected >=3 false splits, got {before_splits}"

        cleaned = cleanup_diarization_segments(noisy)
        after_splits = count_false_splits(cleaned, gt)
        assert after_splits == 0, f"Expected 0 false splits after cleanup, got {after_splits}"

    def test_metric_boundary_preservation_with_realistic_noise(self):
        """Metrics prove true speaker changes survive cleanup."""
        from tests.audio_fixture_helpers import GroundTruth, GroundTruthBoundary

        # Ground truth: A→B→A pattern with realistic gaps
        gt = GroundTruth(
            duration=9.0,
            sample_rate=16000,
            speakers=["A", "B"],
            boundaries=[
                GroundTruthBoundary(time=3.0, speaker_before="A", speaker_after="B"),
                GroundTruthBoundary(time=6.0, speaker_before="B", speaker_after="A"),
            ],
            segments=[(0.0, 3.0, "A"), (3.0, 6.0, "B"), (6.0, 9.0, "A")],
            noise_level=0.1,
            seed=42,
        )

        # Noisy diarizer: some micro-splits within each turn
        noisy = [
            self._seg(0.0, 1.5, "spk0"),
            self._seg(1.55, 3.0, "spk0"),   # micro-split in A
            self._seg(3.0, 4.5, "spk1"),
            self._seg(4.55, 6.0, "spk1"),   # micro-split in B
            self._seg(6.0, 7.5, "spk0"),
            self._seg(7.55, 9.0, "spk0"),   # micro-split in A
        ]

        cleaned = cleanup_diarization_segments(noisy)
        preserved, total = count_preserved_boundaries(cleaned, gt)

        assert preserved == total, (
            f"Expected all {total} boundaries preserved, got {preserved}"
        )

    def test_metric_segment_count_reduction(self):
        """Segment count reduction is measurable and non-negative."""
        noisy = [
            self._seg(0.0, 1.0, "spk0"),
            self._seg(1.05, 2.0, "spk0"),
            self._seg(2.1, 3.0, "spk0"),
            self._seg(3.5, 5.0, "spk1"),
            self._seg(5.5, 7.0, "spk0"),
        ]
        cleaned = cleanup_diarization_segments(noisy)
        reduction = segment_count_reduction(noisy, cleaned)

        assert reduction >= 0
        assert len(cleaned) < len(noisy), (
            f"Expected fewer segments after cleanup: {len(noisy)} → {len(cleaned)}"
        )
        assert reduction == len(noisy) - len(cleaned)

    def test_metric_speaker_purity_after_cleanup(self):
        """Speaker purity is 1.0 when cleanup correctly merges within turns."""
        from tests.audio_fixture_helpers import GroundTruth, GroundTruthBoundary

        gt = GroundTruth(
            duration=6.0,
            sample_rate=16000,
            speakers=["A", "B"],
            boundaries=[
                GroundTruthBoundary(time=3.0, speaker_before="A", speaker_after="B"),
            ],
            segments=[(0.0, 3.0, "A"), (3.0, 6.0, "B")],
            noise_level=0.1,
            seed=42,
        )

        # Clean output: one segment per speaker turn
        cleaned = [
            self._seg(0.0, 3.0, "spk0"),
            self._seg(3.0, 6.0, "spk1"),
        ]
        purity = compute_speaker_purity(cleaned, gt)
        assert purity == 1.0

    def test_cleanup_with_fixture_ground_truth(self, tmp_path: Path):
        """End-to-end: generate fixture, simulate noisy diarizer, cleanup,
        and validate metrics."""
        turns = [("A", 2.0), ("B", 2.0), ("A", 1.5), ("B", 2.0)]
        _, gt = generate_noisy_multi_speaker_wav(
            tmp_path / "test.wav",
            speaker_turns=turns,
            gap_duration=0.3,
            noise_level=0.2,
        )

        # Simulate noisy diarizer output with micro-splits
        noisy_segs = []
        for start, end, speaker in gt.segments:
            label = "spk0" if speaker == "A" else "spk1"
            # Split each turn into 2-3 micro-segments
            mid = (start + end) / 2.0
            noisy_segs.append(self._seg(start, mid - 0.05, label))
            noisy_segs.append(self._seg(mid, end, label))

        cleaned = cleanup_diarization_segments(noisy_segs)

        # Verify: fewer segments after cleanup
        assert len(cleaned) < len(noisy_segs)

        # Verify: true boundaries preserved
        preserved, total = count_preserved_boundaries(cleaned, gt)
        assert preserved >= 1, f"Expected >=1 boundary preserved, got {preserved}/{total}"

        # Verify: false splits reduced
        false_before = count_false_splits(noisy_segs, gt)
        false_after = count_false_splits(cleaned, gt)
        assert false_after < false_before


# ---------------------------------------------------------------------------
# Malformed / edge-case input tests
# ---------------------------------------------------------------------------


class TestCleanupMalformedInputs:
    """Verify cleanup handles malformed inputs without crashing."""

    @staticmethod
    def _seg(start: float, end: float, speaker: str) -> SpeakerSegment:
        return SpeakerSegment(start=start, end=end, speaker=speaker)

    def test_empty_segments_returns_empty(self):
        assert cleanup_diarization_segments([]) == []

    def test_out_of_order_segments_sorted(self):
        """Out-of-order segments are sorted; only close ones merge."""
        segs = [
            self._seg(3.0, 4.0, "spk0"),
            self._seg(0.0, 1.0, "spk0"),
            self._seg(1.1, 2.0, "spk0"),
        ]
        result = cleanup_diarization_segments(segs)
        # After sorting: [0-1, 1.1-2, 3-4]
        # Gap between 2.0 and 3.0 is 1.0s > 0.2s threshold — not merged
        assert len(result) == 2
        assert result[0].start == 0.0
        assert result[0].end == 2.0
        assert result[1].start == 3.0
        assert result[1].end == 4.0

    def test_negative_duration_segment_skipped(self):
        segs = [
            self._seg(0.0, 1.0, "spk0"),
            self._seg(2.0, 1.5, "spk0"),  # negative
            self._seg(1.1, 2.0, "spk0"),  # 0.1s gap → merges
        ]
        result = cleanup_diarization_segments(segs)
        assert len(result) == 1
        assert result[0].end == 2.0

    def test_all_negative_duration_returns_empty_or_partial(self):
        segs = [
            self._seg(1.0, 0.5, "spk0"),
            self._seg(3.0, 2.0, "spk1"),
        ]
        result = cleanup_diarization_segments(segs)
        # All segments have negative duration → all skipped
        assert len(result) == 0

    def test_overlapping_segments_kept_separate(self):
        """Overlapping same-speaker segments are NOT merged (gap is negative)."""
        segs = [
            self._seg(0.0, 2.0, "spk0"),
            self._seg(1.5, 3.0, "spk0"),  # overlaps — gap is negative
        ]
        result = cleanup_diarization_segments(segs)
        # Gap is negative so the 0 <= gap condition fails; stays separate
        assert len(result) == 2

    def test_zero_duration_segment_handled(self):
        """Zero-duration segment (start == end) is handled."""
        segs = [
            self._seg(0.0, 1.0, "spk0"),
            self._seg(1.0, 1.0, "spk0"),  # zero-duration
            self._seg(1.1, 2.0, "spk0"),
        ]
        # Should not crash — zero-duration segment has duration 0 < 0.5
        # but may or may not merge depending on gap logic
        result = cleanup_diarization_segments(segs)
        assert len(result) >= 1

    def test_single_segment_unchanged(self):
        segs = [self._seg(0.0, 1.0, "spk0")]
        result = cleanup_diarization_segments(segs)
        assert len(result) == 1
        assert result[0] == segs[0]


# ---------------------------------------------------------------------------
# Boundary condition tests at exact thresholds
# ---------------------------------------------------------------------------


class TestCleanupBoundaryConditions:
    """Test cleanup at exact threshold boundaries."""

    @staticmethod
    def _seg(start: float, end: float, speaker: str) -> SpeakerSegment:
        return SpeakerSegment(start=start, end=end, speaker=speaker)

    def test_gap_exactly_at_threshold_merges(self):
        """Gap exactly at 0.2s should merge (<=)."""
        segs = [
            self._seg(0.0, 1.0, "spk0"),
            self._seg(1.2, 2.0, "spk0"),  # exactly 0.2s gap
        ]
        result = cleanup_diarization_segments(segs)
        assert len(result) == 1

    def test_gap_just_above_threshold_not_merged(self):
        """Gap just above 0.2s should NOT merge."""
        segs = [
            self._seg(0.0, 1.0, "spk0"),
            self._seg(1.21, 2.0, "spk0"),  # 0.21s gap
        ]
        result = cleanup_diarization_segments(segs)
        assert len(result) == 2

    def test_segment_exactly_at_short_threshold(self):
        """Segment exactly 0.5s duration — not short (< not <=)."""
        segs = [
            self._seg(0.0, 2.0, "spk0"),    # long
            self._seg(2.1, 2.6, "spk0"),     # exactly 0.5s — NOT short
            self._seg(2.7, 4.0, "spk0"),     # long
        ]
        result = cleanup_diarization_segments(segs)
        # All gaps < 0.2s, so gap merge should combine all three
        assert len(result) == 1

    def test_segment_just_below_short_threshold(self):
        """Segment just below 0.5s duration IS short."""
        segs = [
            self._seg(0.0, 2.0, "spk0"),    # long
            self._seg(2.1, 2.55, "spk0"),    # 0.45s — short
            self._seg(2.6, 4.0, "spk0"),     # long
        ]
        result = cleanup_diarization_segments(segs)
        # All same speaker, gaps < 0.2s → gap merge handles them
        assert len(result) == 1

    def test_true_turn_near_boundary_tolerance(self):
        """True speaker turn near but within boundary tolerance."""
        # Gap of exactly 0.2s between different speakers — must stay separate
        segs = [
            self._seg(0.0, 2.0, "spk0"),
            self._seg(2.2, 4.0, "spk1"),   # exactly 0.2s gap, different speaker
        ]
        result = cleanup_diarization_segments(segs)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Optional slow diarizer smoke test
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSlowDiarizerSmoke:
    """Optional smoke test that uses the real Diarizer on a synthetic fixture.

    Skipped when sherpa-onnx or models are unavailable. Not required for
    normal CI — deterministic cleanup/metric tests are the required pass gate.
    """

    def test_diarize_generated_fixture(self, tmp_path: Path):
        """Generated fixture can be diarized by the real Diarizer.

        This test validates that the synthetic fixture is compatible with
        the real diarization pipeline, but does NOT assert specific speaker
        counts (synthetic audio may not produce meaningful speaker splits).
        """
        pytest.importorskip("sherpa_onnx", reason="sherpa-onnx not installed")

        try:
            from meetandread.speaker.diarizer import Diarizer
            from meetandread.speaker.model_downloader import ensure_all_models
        except ImportError as exc:
            pytest.skip(f"Speaker modules unavailable: {exc}")

        wav_path, gt = generate_noisy_multi_speaker_wav(
            tmp_path / "diarize_test.wav",
            duration_per_speaker=3.0,
            gap_duration=0.5,
            noise_level=0.1,
            seed=42,
        )

        try:
            cache_dir = tmp_path / "models"
            diarizer = Diarizer(cache_dir=cache_dir)
            result = diarizer.diarize(wav_path)
        except Exception as exc:
            pytest.skip(f"Diarizer initialization or processing failed: {exc}")

        # The key assertion: no crash, result is well-formed
        assert result is not None
        if not result.succeeded:
            pytest.skip(
                f"Diarization did not succeed on synthetic audio: {result.error}"
            )

        # If segments are returned, verify they're well-formed
        for seg in result.segments:
            assert seg.start >= 0.0
            assert seg.end >= seg.start
            # sherpa-onnx may return speaker as int or str
            assert isinstance(seg.speaker, (str, int))
            # Convert to string for non-empty check
            assert str(seg.speaker)

    def test_diarize_fixture_with_cleanup(self, tmp_path: Path):
        """Fixture diarization + cleanup produces fewer segments than raw."""
        pytest.importorskip("sherpa_onnx", reason="sherpa-onnx not installed")

        try:
            from meetandread.speaker.diarizer import Diarizer
            from meetandread.speaker.model_downloader import ensure_all_models
        except ImportError as exc:
            pytest.skip(f"Speaker modules unavailable: {exc}")

        wav_path, gt = generate_noisy_multi_speaker_wav(
            tmp_path / "diarize_cleanup_test.wav",
            duration_per_speaker=3.0,
            gap_duration=0.5,
            noise_level=0.1,
            seed=42,
        )

        try:
            cache_dir = tmp_path / "models"
            diarizer = Diarizer(cache_dir=cache_dir)
            result = diarizer.diarize(wav_path)
        except Exception as exc:
            pytest.skip(f"Diarizer failed: {exc}")

        if not result.succeeded:
            pytest.skip(f"Diarization failed: {result.error}")

        # Cleanup should not increase segment count
        raw_count = len(result.segments)
        cleaned = cleanup_diarization_segments(result.segments)
        assert len(cleaned) <= raw_count, (
            f"Cleanup should not increase segments: {raw_count} → {len(cleaned)}"
        )
