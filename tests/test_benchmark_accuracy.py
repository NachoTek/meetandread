"""WER benchmark accuracy test.

Runs Whisper base model against real audio with ground truth to measure
actual Word Error Rate. The test always passes — it's measuring, not gating.
Results are saved to a human-readable report file for analysis.

Run with: pytest tests/test_benchmark_accuracy.py -v -m slow
"""

import pytest
import tempfile
from pathlib import Path

from meetandread.transcription.engine import WhisperTranscriptionEngine
from meetandread.performance.benchmark import BenchmarkRunner
from meetandread.performance.wer import calculate_wer_details

# Default test data paths (created by T01)
BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "src" / "meetandread" / "performance" / "test_data"
TEST_CLIP = BENCHMARK_DIR / "benchmark.wav"
GROUND_TRUTH = BENCHMARK_DIR / "benchmark_ground_truth.txt"
RESULTS_FILE = Path(tempfile.gettempdir()) / "meetandread-test-reports" / "benchmark_results.txt"

# Target WER threshold (informational — test passes regardless)
WER_TARGET = 0.05

# ASCII-safe symbols for Windows console compatibility
_CHECK = "[OK]"
_CROSS = "[X]"
_WARN = "[!]"


@pytest.mark.slow
def test_wer_benchmark_against_real_audio():
    """Run Whisper base model against real audio and record WER.

    Loads the benchmark.wav (150s single-speaker clip extracted from
    SAMPLE-Audio1.wav) and its verified ground truth, runs full
    transcription through the Whisper base model, and computes WER.

    The test always passes — it records the measurement. If WER exceeds
    the 5% target, a gap analysis is included in the results report.
    """
    # Verify test data files exist
    assert TEST_CLIP.exists(), f"Benchmark audio not found: {TEST_CLIP}"
    assert GROUND_TRUTH.exists(), f"Ground truth not found: {GROUND_TRUTH}"

    # Load ground truth for reference
    reference_text = GROUND_TRUTH.read_text(encoding="utf-8").strip()
    assert len(reference_text) > 0, "Ground truth is empty"

    # Set up engine and runner
    engine = WhisperTranscriptionEngine(model_size="base")
    engine.load_model()
    assert engine.is_model_loaded(), "Model failed to load"

    runner = BenchmarkRunner(
        engine=engine,
        test_clip_path=TEST_CLIP,
        ground_truth_path=GROUND_TRUTH,
        chunk_duration_s=5.0,
    )

    # Run benchmark synchronously
    result = runner.run()

    # Verify benchmark completed without error
    assert result is not None, "Benchmark returned None"
    assert result.error is None, f"Benchmark failed with error: {result.error}"
    assert result.total_audio_s > 0, "No audio was processed"
    assert len(result.hypothesis_text) > 0, "Transcription produced no text"

    # Record detailed WER
    details = calculate_wer_details(reference_text, result.hypothesis_text)

    # Build and save results report
    report_lines = _build_report(result, details, reference_text)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text("\n".join(report_lines), encoding="utf-8")

    # Log key results for quick visibility
    print(f"\n{'='*60}")
    print(f"WER BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"  WER:             {details.wer:.2%} (target: <= {WER_TARGET:.0%})")
    print(f"  Substitutions:   {details.substitutions}")
    print(f"  Deletions:       {details.deletions}")
    print(f"  Insertions:      {details.insertions}")
    print(f"  Reference words: {details.reference_length}")
    print(f"  Hypothesis words:{details.hypothesis_length}")
    print(f"  Audio duration:  {result.total_audio_s:.1f}s")
    print(f"  Processing time: {result.total_latency_s:.1f}s")
    print(f"  Throughput:      {result.throughput_ratio:.1f}x realtime")
    print(f"  Report saved to: {RESULTS_FILE}")
    if details.wer > WER_TARGET:
        print(f"  {_WARN} WER exceeds {WER_TARGET:.0%} target -- see gap analysis in report")
    print(f"{'='*60}\n")

    # Test always passes — it's measuring, not gating


def _build_report(result, details, reference_text):
    """Build human-readable benchmark results report.

    Args:
        result: BenchmarkResult from the runner.
        details: WERDetail with operation breakdown.
        reference_text: Ground truth text.

    Returns:
        List of report lines.
    """
    lines = [
        "=" * 70,
        "WER BENCHMARK RESULTS",
        "=" * 70,
        "",
        "MODEL INFORMATION",
        "-" * 40,
    ]

    for key, val in result.model_info.items():
        lines.append(f"  {key}: {val}")

    lines.extend([
        "",
        "PERFORMANCE",
        "-" * 40,
        f"  Audio duration:       {result.total_audio_s:.1f}s",
        f"  Processing time:      {result.total_latency_s:.1f}s",
        f"  Throughput:           {result.throughput_ratio:.1f}x realtime",
        f"  Chunks processed:     {len(result.chunk_latencies)}",
    ])

    if result.chunk_latencies:
        avg_latency = sum(c.latency_s for c in result.chunk_latencies) / len(result.chunk_latencies)
        max_latency = max(c.latency_s for c in result.chunk_latencies)
        lines.extend([
            f"  Avg chunk latency:    {avg_latency:.3f}s",
            f"  Max chunk latency:    {max_latency:.3f}s",
        ])

    lines.extend([
        "",
        "WORD ERROR RATE",
        "-" * 40,
        f"  WER:                  {details.wer:.4f} ({details.wer:.2%})",
        f"  Target:               ≤ {WER_TARGET:.0%}",
        f"  Substitutions (S):    {details.substitutions}",
        f"  Deletions (D):        {details.deletions}",
        f"  Insertions (I):       {details.insertions}",
        f"  Reference words (N):  {details.reference_length}",
        f"  Hypothesis words:     {details.hypothesis_length}",
        f"  Error count (S+D+I):  {details.substitutions + details.deletions + details.insertions}",
        f"  Target met:           {'YES ' + _CHECK if details.wer <= WER_TARGET else 'NO ' + _CROSS}",
    ])

    lines.extend([
        "",
        "GROUND TRUTH (first 500 chars)",
        "-" * 40,
    ])
    lines.extend(_wrap_text(reference_text[:500], prefix="  "))

    lines.extend([
        "",
        "HYPOTHESIS (first 500 chars)",
        "-" * 40,
    ])
    lines.extend(_wrap_text(result.hypothesis_text[:500], prefix="  "))

    # Gap analysis if WER exceeds target
    if details.wer > WER_TARGET:
        lines.extend([
            "",
            "=" * 70,
            "GAP ANALYSIS — WER exceeds target",
            "=" * 70,
            "",
            f"  Actual WER:   {details.wer:.2%}",
            f"  Target WER:   {WER_TARGET:.0%}",
            f"  Gap:          {details.wer - WER_TARGET:.2%} above target",
            "",
            "LIKELY CAUSES:",
            "  1. Base model limitations — the 'base' model (74M params) has limited",
            "     capacity for nuanced speech recognition compared to small/medium/large.",
            "  2. Speech clarity — conversational speech with varying pace, volume,",
            "     and enunciation is harder to transcribe than studio-quality audio.",
            "  3. Domain-specific vocabulary — technical terms (e.g., 'Claudebot',",
            "     cost optimization jargon) may not be in the base model's vocabulary.",
            "  4. Audio preprocessing — no noise reduction or normalization applied",
            "     before transcription.",
            "",
            "REMEDIATION RECOMMENDATIONS:",
            "  1. Upgrade to 'small' model (244M params) — typically reduces WER by",
            "     30-50% at ~2x processing cost.",
            "  2. Improve audio preprocessing — add noise gate, normalization, and",
            "     band-pass filtering before transcription.",
            "  3. Apply post-processing — use language model rescoring or text",
            "     normalization to fix systematic errors.",
            "  4. Consider 'medium' model if 'small' is insufficient — further",
            "     20-30% WER reduction at additional processing cost.",
        ])

    lines.extend([
        "",
        "=" * 70,
        "END OF REPORT",
        "=" * 70,
    ])

    return lines


def _wrap_text(text, width=80, prefix=""):
    """Wrap text to width with optional prefix."""
    words = text.split()
    lines = []
    current = prefix
    for word in words:
        if len(current) + len(word) + 1 > width + len(prefix):
            if current.strip():
                lines.append(current)
            current = prefix + word
        else:
            if current == prefix:
                current += word
            else:
                current += " " + word
    if current.strip():
        lines.append(current)
    return lines
