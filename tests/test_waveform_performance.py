"""Waveform visualization performance measurement and assertions.

Measures the CPU cost of the recording waveform pipeline per animation
frame, along with a slower (``slow``-marked) session-level diagnostic
benchmark. Produces metrics-only output — no audio payloads,
transcripts, or secrets.

Load-tolerant redesign (issue #148; quarantine history #142/#147): the
CI-regression test no longer asserts an absolute CPU percent. The
self-hosted CI runner shares the box with local agent and human
workloads, so an absolute threshold cannot distinguish box load from a
real regression. Instead the test measures the real per-frame pipeline
(the exact three production calls ``_update_animations`` makes each
frame: ``feed_audio_for_transcription`` → ``get_live_audio_samples`` →
``set_waveform_samples``) against a frozen known-good reference
implementation of the same algorithm in-process, and asserts the ratio
stays within a budget (``tests/load_tolerant_budget.py``, the shared
helper shape introduced by the #149 sustained-load rework). Uniform box
load slows both lanes roughly equally; a subject-only regression (e.g. a
busy-wait or added O(n) work per frame) inflates only the subject lane
and blows the budget.

A companion meta-test permanently verifies the harness catches a
deliberate CPU regression (injected busy-wait in the render path), and a
memory test guards against per-frame heap growth in the same loop.

Quick local run (CI-regression test included, runs everywhere):
    python -m pytest tests/test_waveform_performance.py -q

Run detailed session benchmark (writes report, ``slow``-marked):
    python -m pytest tests/test_waveform_performance.py -q -m slow
"""

import gc
import math
import tempfile
import threading
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional
from unittest.mock import MagicMock

import numpy as np
import psutil
import pytest

from PyQt6.QtWidgets import QApplication

from meetandread.recording.controller import ControllerState, RecordingController
from meetandread.widgets.main_widget import RecordButtonItem

from tests.load_tolerant_budget import assert_load_tolerant_budget

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "src" / "meetandread" / "performance" / "test_data"
TEST_CLIP = BENCHMARK_DIR / "benchmark.wav"

CPU_TARGET_PERCENT = 10.0  # informational only (session benchmark report)
MEMORY_TARGET_MB = 50.0    # informational only (session benchmark report)

# Default durations (session benchmark)
FAST_DURATION_S = 3.0
SLOW_DURATION_S = 8.0
SAMPLE_INTERVAL_S = 0.1  # CPU sampling interval

# Load-tolerant frame-loop constants (issue #148)
SAMPLE_RATE_HZ = 16000
FRAME_SAMPLE_COUNT = 1024        # matches the fake-source capture blocksize
WAVEFORM_POLL_WINDOW_S = 0.1     # matches _update_animations' poll window
WARMUP_FRAMES = 200              # fill the 12s rolling buffer to steady state
MEASURE_FRAMES = 1200
LANE_REPS = 3
MEMORY_PROBE_FRAMES = 300
MEMORY_BOUND_MB = 4.0
META_FRAMES = 200                # regression-detection meta-test lane size
META_LANE_REPS = 2
INJECTED_SLOWDOWN_S = 0.001      # artificial per-frame busy-wait for the meta-test

# Frozen reference snapshot of the healthy per-frame algorithm
REFERENCE_TARGET_POINTS = 60     # RecordButtonItem._WAVEFORM_TARGET_POINTS
REFERENCE_FLAT_AMPLITUDE = 0.005
REFERENCE_DECAY = 0.90
REFERENCE_MAX_BUFFER_BYTES = 12 * 16000 * 2  # controller._live_max_buffer_bytes

# ASCII-safe symbols for Windows console compatibility
_CHECK = "[OK]"
_CROSS = "[X]"
_WARN = "[!]"


# ---------------------------------------------------------------------------
# Measurement data (session benchmark)
# ---------------------------------------------------------------------------

@dataclass
class PerformanceResult:
    """Container for waveform performance measurement results."""
    duration_s: float
    cpu_samples: List[float] = field(default_factory=list)
    avg_cpu_percent: float = 0.0
    peak_cpu_percent: float = 0.0
    peak_heap_mb: float = 0.0
    diagnostics: Optional[dict] = None
    cpu_target: float = CPU_TARGET_PERCENT
    memory_target: float = MEMORY_TARGET_MB
    cpu_pass: bool = False
    memory_pass: bool = False

    def __post_init__(self):
        valid = [s for s in self.cpu_samples if math.isfinite(s)]
        if valid:
            self.avg_cpu_percent = sum(valid) / len(valid)
            self.peak_cpu_percent = max(valid)
        self.cpu_pass = self.avg_cpu_percent < self.cpu_target
        self.memory_pass = self.peak_heap_mb < self.memory_target


# ---------------------------------------------------------------------------
# Shared measurement helper (session benchmark)
# ---------------------------------------------------------------------------

def _run_waveform_performance_measurement(
    duration_s: float = FAST_DURATION_S,
    report_path: Optional[Path] = None,
    clip_path: Optional[Path] = None,
) -> PerformanceResult:
    """Run a waveform recording session and measure CPU / memory overhead.

    Uses the tracked ``benchmark.wav`` fixture as a fake audio source with
    ``RecordingController(enable_transcription=False)``. CPU is sampled via
    ``psutil.Process().cpu_percent(interval=None)`` in a background thread;
    memory is tracked with ``tracemalloc``.

    Report-only: results feed the ``slow``-marked diagnostic benchmark and
    its persisted report. No lane asserts on the absolute CPU number —
    that assert was the quarantined flake (issues #142/#148).

    Args:
        duration_s: Wall-clock recording duration in seconds.
        report_path: If provided, write a detailed text report to this path.
        clip_path: Override the test clip path (defaults to TEST_CLIP).

    Returns:
        PerformanceResult with CPU samples, averages, peak heap, and pass/fail.

    Raises:
        FileNotFoundError: If the benchmark WAV fixture is missing.
        ValueError: If duration_s is not positive.
        RuntimeError: If no valid CPU samples were collected.
    """
    # Resolve the audio source path
    audio_source = clip_path or TEST_CLIP

    # --- Input validation ---
    if not audio_source.exists():
        raise FileNotFoundError(f"Benchmark audio not found: {audio_source}")
    if duration_s <= 0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")

    cpu_samples: List[float] = []
    stop_event = threading.Event()
    proc = psutil.Process()
    cpu_count = psutil.cpu_count() or 1  # Normalize multi-thread CPU to system %

    controller = RecordingController(enable_transcription=False)

    # --- CPU sampling thread ---
    def _sample_cpu():
        # Prime psutil's internal baseline
        proc.cpu_percent(interval=None)
        while not stop_event.is_set():
            # Small sleep to avoid busy-wait; then read non-blocking
            stop_event.wait(SAMPLE_INTERVAL_S)
            if stop_event.is_set():
                break
            try:
                # Normalize per-process CPU by logical core count so that
                # multi-threaded audio I/O on a 12-core machine reports ~9%
                # instead of ~110% (which is 9% of total system capacity).
                val = proc.cpu_percent(interval=None) / cpu_count
                if math.isfinite(val):
                    cpu_samples.append(val)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

    diag: Optional[dict] = None
    try:
        # Start tracemalloc around the measurement window only.
        # Capture a baseline snapshot so we can report the DELTA (waveform
        # overhead) rather than the total process heap which includes import-
        # time allocations from NumPy / PyQt6 / sherpa-onnx that are not
        # attributable to the waveform pipeline.
        tracemalloc.start()
        baseline_current, _ = tracemalloc.get_traced_memory()

        # Start CPU sampling
        sampler_thread = threading.Thread(target=_sample_cpu, daemon=True, name="CpuSampler")
        sampler_thread.start()

        # Start recording with fake source (loop for sustained measurement)
        err = controller.start({"fake"}, fake_path=str(audio_source), fake_loop=True)
        if err:
            raise RuntimeError(f"Controller start failed: {err.message}")

        # Let it run for the target duration
        time.sleep(duration_s)

    finally:
        # Always stop controller and join threads
        try:
            controller.stop()
            # Wait for stop worker to finish (bounded)
            wt = controller._worker_thread
            if wt is not None:
                wt.join(timeout=5.0)
        except Exception:
            pass

        stop_event.set()
        sampler_thread.join(timeout=2.0)

        # Capture diagnostics defensively
        try:
            diag = controller.get_diagnostics()
        except Exception:
            diag = None

        # Read peak memory delta before stopping tracemalloc
        peak_heap_mb = 0.0
        try:
            if tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                # Use current delta over baseline as the overhead metric.
                # The tracemalloc "peak" is a high-water mark since start()
                # and includes baseline allocations — using current-baseline
                # gives the actual waveform pipeline overhead.
                heap_delta = max(0, current - baseline_current)
                peak_heap_mb = heap_delta / (1024 * 1024)
        except Exception:
            pass
        finally:
            try:
                tracemalloc.stop()
            except Exception:
                pass

    # --- Build result ---
    if not cpu_samples:
        raise RuntimeError(
            f"No valid CPU samples collected over {duration_s:.1f}s. "
            f"Controller diagnostics: {diag}"
        )

    result = PerformanceResult(
        duration_s=duration_s,
        cpu_samples=cpu_samples,
        peak_heap_mb=peak_heap_mb,
        diagnostics=diag,
    )

    # --- Write report if requested ---
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = _build_report(result)
        report_path.write_text("\n".join(lines), encoding="utf-8")

    return result


# ---------------------------------------------------------------------------
# Qt application fixture (session-scoped)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    """Provide a QApplication for the test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


# ---------------------------------------------------------------------------
# Load-tolerant frame-loop lanes (issue #148)
# ---------------------------------------------------------------------------

def _mock_parent() -> MagicMock:
    """Parent mock matching the RecordButtonItem test convention."""
    parent = MagicMock()
    parent.is_dragging = False
    parent._click_consumed = False
    return parent


def _synth_chunks(count: int, seed: int) -> List[np.ndarray]:
    """Deterministic float32 audio chunks (speech-like amplitude scale)."""
    rng = np.random.default_rng(seed)
    return [
        (rng.standard_normal(FRAME_SAMPLE_COUNT) * 0.3).astype(np.float32)
        for _ in range(count)
    ]


def _make_subject_frame(
    controller: RecordingController, button: RecordButtonItem
) -> Callable[[np.ndarray], None]:
    """Bind one production waveform frame: feed -> snapshot -> render.

    This is the exact per-frame call sequence ``_update_animations``
    issues every animation frame while recording
    (widgets/main_widget.py): the audio consumer thread feeds PCM into
    the live buffer, then the UI polls a 0.1s snapshot and hands it to
    the render path. Driving the three real production methods directly
    keeps threads (and their scheduler noise) out of the measurement
    while measuring the real code a regression would land in.
    """
    controller._state = ControllerState.RECORDING  # gate for feed buffering

    def _frame(chunk: np.ndarray) -> None:
        controller.feed_audio_for_transcription(chunk)
        samples = controller.get_live_audio_samples(
            duration_seconds=WAVEFORM_POLL_WINDOW_S
        )
        button.set_waveform_samples(samples)

    return _frame


class FrozenReferenceWaveform:
    """Known-good snapshot of the per-frame waveform pipeline algorithm.

    Cost reference for the load-tolerant CPU budget (issue #148). Each
    ``frame(chunk)`` replays the dominant per-frame work of the healthy
    pipeline as of this snapshot:

    - producer: float32 → int16 PCM conversion, locked rolling-buffer
      append + trim (``RecordingController.feed_audio_for_transcription``)
    - consumer: locked buffer snapshot, int16 → float32 renormalization,
      defensive copy (``RecordingController.get_live_audio_samples``)
    - renderer: per-point decay, bounded downsample to 60 points via
      ``linspace`` indexing, peak-hold merge
      (``RecordButtonItem.set_waveform_samples``)

    Deliberately frozen rather than a call-through: the reference must
    stay a known-good baseline, so a regression introduced into any of
    the production methods inflates the subject/reference ratio instead
    of silently slowing the baseline along with it.
    """

    def __init__(self) -> None:
        self._data: List[float] = [REFERENCE_FLAT_AMPLITUDE] * REFERENCE_TARGET_POINTS
        self._buffer = bytearray()
        self._lock = threading.Lock()

    def frame(self, chunk: np.ndarray) -> None:
        # -- producer half: float32 -> int16 PCM + rolling-buffer append --
        clamped = np.clip(chunk, -1.0, 1.0)
        pcm_int16 = (clamped * 32767).astype(np.int16)
        pcm_bytes = pcm_int16.tobytes()
        with self._lock:
            self._buffer.extend(pcm_bytes)
            if len(self._buffer) > REFERENCE_MAX_BUFFER_BYTES:
                excess = len(self._buffer) - REFERENCE_MAX_BUFFER_BYTES
                del self._buffer[:excess]

        # -- consumer half: snapshot + tail slice + renormalize + copy --
        requested_bytes = int(WAVEFORM_POLL_WINDOW_S * SAMPLE_RATE_HZ * 2)
        with self._lock:
            buf_snapshot = bytes(self._buffer) if self._buffer else b""
        available = min(requested_bytes, len(buf_snapshot))
        raw = buf_snapshot[-available:]
        aligned_len = (len(raw) // 2) * 2
        pcm = np.frombuffer(raw[:aligned_len], dtype=np.int16)
        samples = (pcm.astype(np.float32) / 32768.0).copy()

        # -- render half: decay + bounded downsample + peak-hold --
        flat = REFERENCE_FLAT_AMPLITUDE
        self._data = [
            flat + (v - flat) * REFERENCE_DECAY for v in self._data
        ]
        arr = np.asarray(samples, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.flatten()
        arr = np.where(np.isfinite(arr), arr, 0.0)
        if arr.size == 0:
            return
        if arr.size <= REFERENCE_TARGET_POINTS:
            new_vals = np.abs(arr).tolist()
            new_vals.extend([flat] * (REFERENCE_TARGET_POINTS - len(new_vals)))
            new_vals = new_vals[:REFERENCE_TARGET_POINTS]
        else:
            indices = np.linspace(0, arr.size - 1, REFERENCE_TARGET_POINTS, dtype=int)
            new_vals = np.abs(arr[indices]).tolist()
        self._data = [
            max(new_vals[i], self._data[i]) for i in range(REFERENCE_TARGET_POINTS)
        ]


def _timed_frames(frame: Callable[[np.ndarray], None], chunks: List[np.ndarray]) -> float:
    """Time one lane pass over ``chunks`` (GC settled, monotonic clock)."""
    gc.collect()
    start = time.perf_counter()
    for chunk in chunks:
        frame(chunk)
    return time.perf_counter() - start


def _measure_frame_lanes(
    subject_frame: Callable[[np.ndarray], None],
    reference: FrozenReferenceWaveform,
    chunks: List[np.ndarray],
    reps: int,
) -> "tuple[float, float]":
    """Interleave subject/reference passes; return best time of each.

    Interleaving minimizes the window in which box load arriving mid-test
    could slow one lane only; the minimum over reps absorbs transient
    spikes (see ``tests/load_tolerant_budget.py``).
    """
    subject_times: List[float] = []
    reference_times: List[float] = []
    for _ in range(reps):
        subject_times.append(_timed_frames(subject_frame, chunks))
        reference_times.append(_timed_frames(reference.frame, chunks))
    return min(subject_times), min(reference_times)


def _warm_up_lanes(
    subject_frame: Callable[[np.ndarray], None],
    reference: FrozenReferenceWaveform,
    frames: int = WARMUP_FRAMES,
) -> None:
    """Drive both lanes to steady state (full 12s rolling buffer)."""
    warmup_chunks = _synth_chunks(frames, seed=7)
    for chunk in warmup_chunks:
        subject_frame(chunk)
        reference.frame(chunk)


# ---------------------------------------------------------------------------
# CI regression test (load-tolerant, runs in every default lane)
# ---------------------------------------------------------------------------

def test_waveform_performance_ci_regression(qapp):
    """Waveform frame pipeline stays within budget of a frozen reference.

    Load-tolerant replacement for the quarantined absolute-CPU assert
    (issues #142/#147/#148). Measures the real production frame path
    (feed → snapshot → render) against the frozen known-good reference
    in-process and asserts the ratio stays within the shared budget:
    uniform box load scales both lanes roughly equally, a subject-only
    regression does not.
    """
    controller = RecordingController(enable_transcription=False)
    button = RecordButtonItem(_mock_parent())
    reference = FrozenReferenceWaveform()
    subject_frame = _make_subject_frame(controller, button)

    _warm_up_lanes(subject_frame, reference)

    chunks = _synth_chunks(MEASURE_FRAMES, seed=11)
    subject_s, reference_s = _measure_frame_lanes(
        subject_frame, reference, chunks, reps=LANE_REPS
    )

    per_frame_us = subject_s / MEASURE_FRAMES * 1e6
    ratio = subject_s / reference_s
    print(f"\n{'=' * 50}")
    print("WAVEFORM FRAME PIPELINE (load-tolerant CI regression)")
    print(f"{'=' * 50}")
    print(f"  Frames measured:  {MEASURE_FRAMES} x {LANE_REPS} reps (best of)")
    print(f"  Subject lane:     {subject_s:.4f}s ({per_frame_us:.1f} us/frame)")
    print(f"  Reference lane:   {reference_s:.4f}s")
    print(f"  Ratio:            {ratio:.2f}x (budget: 2.0x)")
    print(f"{'=' * 50}\n")

    assert_load_tolerant_budget(
        subject_s,
        reference_s,
        label="waveform frame pipeline (feed -> snapshot -> render)",
    )


def test_waveform_cpu_regression_is_detected(qapp, monkeypatch):
    """The budget must FAIL when the render path genuinely regresses.

    Permanent guard for the redesign's detection claim (issue #148
    acceptance): a deliberate busy-wait injected into the render path
    must blow the budget, proving the harness catches real per-frame CPU
    regressions rather than only passing on healthy code.
    """
    original = RecordButtonItem.set_waveform_samples

    def _slowed_set_waveform_samples(self, samples) -> None:
        deadline = time.perf_counter() + INJECTED_SLOWDOWN_S
        while time.perf_counter() < deadline:
            pass  # burn CPU: simulates added per-frame work
        original(self, samples)

    monkeypatch.setattr(
        RecordButtonItem, "set_waveform_samples", _slowed_set_waveform_samples
    )

    controller = RecordingController(enable_transcription=False)
    button = RecordButtonItem(_mock_parent())
    reference = FrozenReferenceWaveform()
    subject_frame = _make_subject_frame(controller, button)

    _warm_up_lanes(subject_frame, reference)

    chunks = _synth_chunks(META_FRAMES, seed=11)
    subject_s, reference_s = _measure_frame_lanes(
        subject_frame, reference, chunks, reps=META_LANE_REPS
    )

    with pytest.raises(AssertionError, match="budget"):
        assert_load_tolerant_budget(
            subject_s,
            reference_s,
            label="waveform frame pipeline with injected render slowdown",
        )


def test_waveform_frame_loop_memory_bounded(qapp):
    """Steady-state frame loop must not grow the Python heap.

    Companion to the CPU budget (issue #148): the rolling buffer is a
    fixed 384 KB window and the render state is 60 floats, so sustained
    frame processing must not accumulate allocations. A per-frame leak
    (e.g. a retained snapshot) blows this bound quickly. Load-independent
    by construction — allocation behavior does not depend on box load.
    """
    controller = RecordingController(enable_transcription=False)
    button = RecordButtonItem(_mock_parent())
    subject_frame = _make_subject_frame(controller, button)

    _warm_up_lanes(subject_frame, reference=FrozenReferenceWaveform())

    probe_chunks = _synth_chunks(MEMORY_PROBE_FRAMES, seed=13)
    gc.collect()
    tracemalloc.start()
    baseline_current, _ = tracemalloc.get_traced_memory()
    try:
        for chunk in probe_chunks:
            subject_frame(chunk)
        current, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    heap_delta_mb = max(0, current - baseline_current) / (1024 * 1024)
    print(
        f"\nWaveform frame-loop heap delta over {MEMORY_PROBE_FRAMES} frames: "
        f"{heap_delta_mb:.3f} MB (bound: {MEMORY_BOUND_MB} MB)"
    )
    assert heap_delta_mb < MEMORY_BOUND_MB, (
        f"Frame loop grew the heap by {heap_delta_mb:.3f} MB over "
        f"{MEMORY_PROBE_FRAMES} frames — looks like a per-frame leak in "
        "the waveform pipeline (feed/snapshot/render)."
    )


# ---------------------------------------------------------------------------
# Slow detailed session benchmark (report-only)
# ---------------------------------------------------------------------------

RESULTS_FILE = Path(tempfile.gettempdir()) / "meetandread-test-reports" / "waveform_performance_results.txt"


@pytest.mark.slow
def test_waveform_performance_detailed():
    """Detailed session benchmark: longer duration, persisted report.

    Runs the recording session for SLOW_DURATION_S seconds and writes a
    comprehensive metrics report to ``<OS temp dir>/meetandread-test-reports/``.
    Report-only by design: absolute CPU numbers on a shared box are
    diagnostics, not gates (issue #148).
    """
    result = _run_waveform_performance_measurement(
        duration_s=SLOW_DURATION_S,
        report_path=RESULTS_FILE,
    )

    print(f"\n{'=' * 60}")
    print("WAVEFORM PERFORMANCE (detailed benchmark)")
    print(f"{'=' * 60}")
    print(f"  Duration:        {result.duration_s:.1f}s")
    print(f"  CPU samples:     {len(result.cpu_samples)}")
    print(f"  Avg CPU:         {result.avg_cpu_percent:.1f}% (target: < {CPU_TARGET_PERCENT}%)")
    print(f"  Peak CPU:        {result.peak_cpu_percent:.1f}%")
    print(f"  Peak heap:       {result.peak_heap_mb:.1f} MB (target: < {MEMORY_TARGET_MB} MB)")
    print(f"  CPU pass:        {_CHECK if result.cpu_pass else _CROSS}")
    print(f"  Memory pass:     {_CHECK if result.memory_pass else _CROSS}")
    print(f"  Report saved to: {RESULTS_FILE}")
    print(f"{'=' * 60}\n")

    if not result.cpu_pass:
        gap = result.avg_cpu_percent - CPU_TARGET_PERCENT
        print(
            f"  {_WARN} Avg CPU {result.avg_cpu_percent:.1f}% exceeds "
            f"{CPU_TARGET_PERCENT}% target (gap: +{gap:.1f}%)"
        )

    if not result.memory_pass:
        gap = result.peak_heap_mb - MEMORY_TARGET_MB
        print(
            f"  {_WARN} Peak heap {result.peak_heap_mb:.1f} MB exceeds "
            f"{MEMORY_TARGET_MB} MB target (gap: +{gap:.1f} MB)"
        )


# ---------------------------------------------------------------------------
# Negative / edge-case tests
# ---------------------------------------------------------------------------

def test_measurement_refuses_missing_benchmark_wav(tmp_path):
    """Helper must fail clearly when benchmark.wav is missing."""
    bad_path = tmp_path / "nonexistent.wav"
    assert not bad_path.exists(), "Precondition: file must not exist"
    with pytest.raises(FileNotFoundError, match="Benchmark audio not found"):
        _run_waveform_performance_measurement(duration_s=1.0, clip_path=bad_path)


def test_measurement_refuses_zero_duration():
    """Helper must fail for non-positive duration."""
    with pytest.raises(ValueError, match="duration_s must be positive"):
        _run_waveform_performance_measurement(duration_s=0)


def test_measurement_refuses_negative_duration():
    """Helper must fail for negative duration."""
    with pytest.raises(ValueError, match="duration_s must be positive"):
        _run_waveform_performance_measurement(duration_s=-1.0)


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def _build_report(result: PerformanceResult) -> List[str]:
    """Build a metrics-only performance report (no audio/transcripts/secrets)."""
    lines = [
        "=" * 70,
        "WAVEFORM PERFORMANCE RESULTS",
        "=" * 70,
        "",
        "TEST CONFIGURATION",
        "-" * 40,
        "  Audio source:      benchmark.wav (fake, looped)",
        "  Transcription:     disabled",
        f"  Duration:          {result.duration_s:.1f}s",
        f"  CPU sample count:  {len(result.cpu_samples)}",
        f"  CPU sample interval: {SAMPLE_INTERVAL_S}s",
        "",
        "CPU UTILIZATION",
        "-" * 40,
        f"  Average CPU:       {result.avg_cpu_percent:.1f}%",
        f"  Peak CPU:          {result.peak_cpu_percent:.1f}%",
        f"  Target:            < {result.cpu_target}%",
        f"  Target met:        {'YES ' + _CHECK if result.cpu_pass else 'NO ' + _CROSS}",
        "",
        "MEMORY (Python heap via tracemalloc)",
        "-" * 40,
        f"  Peak heap:         {result.peak_heap_mb:.1f} MB",
        f"  Target:            < {result.memory_target} MB",
        f"  Target met:        {'YES ' + _CHECK if result.memory_pass else 'NO ' + _CROSS}",
        "",
        "NOTE (issue #148)",
        "-" * 40,
        "  Absolute CPU numbers on the shared self-hosted box are",
        "  diagnostics only. Regression gating lives in the load-tolerant",
        "  frame-loop ratio test (tests/load_tolerant_budget.py).",
    ]

    # Controller diagnostics (sanitized — no raw audio/transcripts)
    if result.diagnostics:
        lines.extend([
            "",
            "CONTROLLER DIAGNOSTICS",
            "-" * 40,
        ])
        for key, val in result.diagnostics.items():
            if isinstance(val, dict):
                lines.append(f"  {key}:")
                for k2, v2 in val.items():
                    lines.append(f"    {k2}: {v2}")
            else:
                lines.append(f"  {key}: {val}")

    lines.extend([
        "",
        "=" * 70,
        "END OF REPORT",
        "=" * 70,
    ])

    return lines
