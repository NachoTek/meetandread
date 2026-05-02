# Noisy-Audio Validation Evidence

> End-to-end validation that assembled denoising, VAD-backed live processing, and diarization cleanup work together on deterministic multi-speaker audio. Source of truth is the test suite — this report cites only tracked test files and executable commands.

## Validation Commands

All commands are runnable from a clean checkout with no preexisting generated fixture files:

```bash
# S03 slice verification (56 tests, ~12s)
python -m pytest tests/test_noisy_diarization_validation.py tests/test_recording_controller_noisy_audio_e2e.py -q

# S01/S02 regression: denoising and session (66 tests, ~16s)
python -m pytest tests/test_audio_session.py tests/test_audio_session_denoising.py tests/test_audio_denoising.py -q

# Speaker pipeline, signatures, VAD, config regression (176 tests, ~7s)
python -m pytest tests/test_speaker_pipeline.py tests/test_speaker_signatures.py tests/test_accumulating_processor_vad.py tests/test_config.py -q
```

**Total: 298 tests, all passing.**

## Fixture Design

The synthetic fixtures are generated at test runtime by `tests/audio_fixture_helpers.py`. No committed binary WAVs are needed — every test call produces a deterministic file under `tmp_path`.

| Parameter | Default | Notes |
|-----------|---------|-------|
| Sample rate | 16,000 Hz | 16-bit mono PCM |
| Speaker profiles | A (300/1800/2600 Hz formants), B (600/2200/3200 Hz) | Distinct spectral shapes, not simple tones |
| RNG seed | 42 | Fixed for determinism; same seed always produces identical WAV bytes |
| Gap noise | RMS 0.15 | Gaussian noise during inter-turn silence |
| Default turns | A(2s) → B(2s) → A(1.6s) → B(2s), 0.3s gaps | Total ~8.5s, 4 ground-truth segments, 3 speaker-change boundaries |
| Background noise | 30% of gap noise level overlaid on speech chunks | Simulates real-world noisy recording |

Ground-truth metadata (`GroundTruth` dataclass) carries exact segment boundaries, speaker-change times, and the seed/noise parameters — enabling precise metric computation without manual annotation.

### Design Rationale

- **Formant profiles** rather than single-frequency tones: spectrally richer signals exercise denoising and speaker-distinction logic more realistically.
- **Inline generation** under `tmp_path`: no committed binary fixtures bloating the repo; reproducible from any clean checkout.
- **Parameterized**: custom turns, durations, noise levels, and seeds via keyword arguments for targeted boundary-condition tests.

## False split reduction

Diarization cleanup (`cleanup_diarization_segments` in `src/meetandread/speaker/diarizer.py`) is a conservative post-processing step applied before speaker matching. It targets two patterns common in noisy diarizer output:

| Pattern | Threshold | Action |
|---------|-----------|--------|
| Same-speaker micro-gaps | Gap ≤ 0.2s (`DEFAULT_GAP_MERGE_THRESHOLD`) | Merge into one segment |
| Spurious short same-speaker fragments | Duration < 0.5s (`DEFAULT_SHORT_SEGMENT_THRESHOLD`) | Absorb into adjacent same-speaker segment if doing so does not merge distinct alternating speakers |

### Metric Evidence

The test suite in `test_noisy_diarization_validation.py` uses four metric helpers to objectively measure both sides of the tradeoff:

| Metric | What it measures | Test class |
|--------|-----------------|------------|
| `count_false_splits()` | Segment boundaries inside a single ground-truth speaker turn (not near a real change) | `TestCleanupMetrics` |
| `count_preserved_boundaries()` | Ground-truth speaker changes with a matching cleaned segment boundary | `TestCleanupMetrics` |
| `segment_count_reduction()` | Raw segment count minus post-cleanup count | `TestCleanupMetrics` |
| `compute_speaker_purity()` | Fraction of cleaned segments wholly within a ground-truth turn | `TestCleanupMetrics` |

**Results (asserted by tests):**

- **Micro-split merging:** 4 noisy micro-segments of the same speaker with 0.05–0.1s gaps → 1 merged segment. False splits drop from ≥3 to 0.
- **Segment count reduction:** 5 noisy segments → fewer after cleanup; reduction is always non-negative.
- **Threshold stability:** `DEFAULT_GAP_MERGE_THRESHOLD == 0.2` and `DEFAULT_SHORT_SEGMENT_THRESHOLD == 0.5` are asserted by name in `TestCleanupThresholds` — future tuning changes fail loudly.

### Boundary Conditions

`TestCleanupBoundaryConditions` exercises exact threshold values:

| Condition | Gap/Duration | Result |
|-----------|-------------|--------|
| Gap exactly at threshold | 0.20s | Merged |
| Gap just above threshold | 0.21s | Not merged |
| Segment at short threshold | 0.50s | Not treated as short |
| Segment just below short threshold | 0.45s | Treated as short |
| Different speakers at threshold gap | 0.20s gap, different labels | Kept separate |

## True Speaker Change Preservation

Cleanup must not hide real speaker transitions. The test suite validates this with A/B/A patterns and realistic noisy diarizer output:

- **A/B/A preservation:** 3 segments (spk0 → spk1 → spk0, 0.5s gaps) remain 3 distinct segments after cleanup (`test_true_aba_turns_preserved`).
- **Boundary preservation with noise:** 6 noisy segments (2 micro-splits per speaker turn in an A/B/A pattern) → all ground-truth boundaries preserved after cleanup (`test_metric_boundary_preservation_with_realistic_noise`).
- **End-to-end with fixture:** Generated fixture with 4 ground-truth turns, simulated noisy diarizer output (2 micro-segments per turn), cleanup produces fewer segments while preserving ≥1 true boundary (`test_cleanup_with_fixture_ground_truth`).
- **Speaker purity:** After cleanup, all cleaned segments lie within ground-truth turns → purity = 1.0 (`test_metric_speaker_purity_after_cleanup`).

## FakeAudioModule RecordingController Path

RecordingController was extended in T02 to accept deterministic fake audio sources, enabling full-pipeline E2E testing without real audio devices, Whisper models, or sherpa-onnx.

### API

The `start()` method accepts keyword-only arguments for fake source control:

| Parameter | Type | Effect |
|-----------|------|--------|
| `fake_path` | `str` | Path to WAV file for `FakeAudioModule` to replay |
| `fake_denoise` | `bool` | Whether denoising is enabled for the fake source |
| `fake_loop` | `bool` | Whether the fake source loops the WAV |

Existing mic/system callers are unchanged — fake support is additive.

### Controller E2E Evidence

`test_recording_controller_noisy_audio_e2e.py` contains 18 tests across 6 classes:

| Test class | Count | What is verified |
|-----------|-------|-----------------|
| `TestRecordingControllerNoisyAudioE2E` | 4 | Recording produces WAV output, denoising processes frames with zero fallbacks, controller returns to IDLE |
| `TestRecordingControllerDiarization` | 3 | Cleanup reduces over-segmented count, speaker labels applied to transcript words, failed diarization leaves words unlabeled |
| `TestRecordingControllerNegativePaths` | 6 | Missing path, nonexistent path, no-denoise flag, empty sources, invalid source type, mic/system behavior unchanged |
| `TestRecordingControllerDiagnostics` | 2 | Diagnostics return controller state, no raw audio or embeddings exposed |
| `TestRecordingControllerBoundaryConditions` | 3 | Words inside/outside segments get correct labels, short recording with no segments leaves words unlabeled |

### Denoising Integration

- Fake source with `fake_denoise=True` produces at least one processed frame, zero fallbacks in the happy path.
- Fake source with `fake_denoise=False` processes zero frames through denoising.
- Denoising stats (processed_frame_count, fallback_count, enabled, active) are inspectable via `get_diagnostics()`.

### Diarization Integration

- Over-segmented diarization result (3 sub-segments per ground-truth turn + spurious short segments) is reduced by cleanup.
- Speaker labels are applied to transcript words via midpoint-in-segment matching.
- Failed diarization (empty segments, error message) leaves words unlabeled without crashing.

## Diagnostics and Redaction

### Sanitized Diagnostics Getter

`RecordingController.get_diagnostics()` returns a structured dict containing:

- **Controller state** (IDLE, RECORDING, etc.)
- **Recording/transcript paths** (file system paths only)
- **Session stats** including denoising (processed_frame_count, fallback_count, enabled, active, provider)
- **Transcript word counts** (with and without speaker labels)
- **Diarization metadata** (segment_count, speaker_count, succeeded, error)
- **Error info** (non-recoverable flag, message)

### What Is Excluded

The diagnostics getter intentionally excludes:

- Raw audio samples or waveform data
- Transcript text content
- Speaker embeddings or voice signatures
- Secrets, API keys, or model paths
- Any personally identifiable meeting content

### Logging Surface

All denoising and diarization log events use sanitized content (error classes, frame counts, segment counts) — never raw audio, transcript text, or embeddings. Logs are emitted to `meetandread.audio.session` and `meetandread.recording.controller` loggers.

## Optional Real Diarization Evidence

Two tests in `TestSlowDiarizerSmoke` (marked `@pytest.mark.slow`) exercise the real sherpa-onnx `Diarizer` on generated fixtures:

- **Availability:** These tests skip automatically when sherpa-onnx or speaker models are unavailable.
- **Assertions:** No specific speaker count is asserted (synthetic audio may not produce meaningful speaker splits). Tests verify no crash and well-formed output only.
- **Cleanup validation:** Real diarization + cleanup produces ≤ the raw segment count.

These tests are **optional evidence** — the deterministic cleanup/metric tests (38 in `test_noisy_diarization_validation.py`) are the required pass gate. Real diarization remains valuable but is dependency/runtime-heavy.

## Malformed Input and Error Path Coverage

The test suite covers error paths and malformed inputs at every level:

| Level | Tests | Coverage |
|-------|-------|----------|
| Fixture generation | 4 | Invalid sample rate, negative gap, negative noise, empty turns |
| Cleanup malformed inputs | 6 | Empty lists, out-of-order, negative-duration, zero-duration, overlapping, single segments |
| Cleanup boundary conditions | 5 | Exact threshold, just above/below, true turn near tolerance |
| Controller negative paths | 6 | Missing path, nonexistent path, no-denoise flag, empty sources, invalid type, mic/system unchanged |
| Controller boundary conditions | 3 | Words inside/outside segments, short recording no segments |
| Diarization failure | 1 | Failed result leaves words unlabeled |

## Clean Checkout Guarantee

All tests work from a clean checkout with no preexisting generated fixture files:

- Fixtures are generated under pytest's `tmp_path` (system temp directory, cleaned automatically).
- No tests import from `.gsd/`, `.planning/`, `.audits/`, or other `.gitignore` paths.
- No tests depend on committed binary WAV files.
- The only required dependencies are `numpy`, `pytest`, and the project's own `src/meetandread/` modules.

## Test File Index

| File | Tests | Purpose |
|------|-------|---------|
| `tests/audio_fixture_helpers.py` | — | Fixture generator and validation helper |
| `tests/test_noisy_diarization_validation.py` | 38 | Fixture generation, cleanup metrics, malformed inputs, boundary conditions, optional real diarizer |
| `tests/test_recording_controller_noisy_audio_e2e.py` | 18 | Full controller pipeline with fake source, denoising, diarization, diagnostics |
| `tests/test_audio_denoising.py` | — | Denoising provider unit tests |
| `tests/test_audio_session_denoising.py` | — | AudioSession denoising integration tests |
| `tests/test_audio_session.py` | — | AudioSession core tests |
