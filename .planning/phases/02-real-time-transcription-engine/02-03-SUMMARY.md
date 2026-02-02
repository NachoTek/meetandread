---
phase: 02-real-time-transcription-engine
plan: 03
subsystem: transcription
tags: [whisper, confidence, psutil, hardware-detection, model-recommendation]

# Dependency graph
requires:
  - phase: 02-02
    provides: "ConfigManager for saving hardware recommendations"
provides:
  - "Confidence normalization from Whisper log_probs to 0-100 scale"
  - "Color-coded confidence levels (green/yellow/orange/red)"
  - "Visual distortion effect calculation for low confidence text"
  - "Hardware detection with RAM, CPU, frequency via psutil"
  - "Model size recommendation algorithm (tiny/base/small)"
  - "Config integration for persisting recommendations"
affects:
  - "Phase 5: Widget UI (confidence color coding, legend display)"
  - "Phase 3: Enhancement trigger based on confidence threshold"

# Tech tracking
tech-stack:
  added: [psutil]
  patterns: 
    - "Confidence normalization: linear mapping from log_prob ranges"
    - "Hardware detection: 60-second caching with refresh()"
    - "Model recommendation: threshold-based algorithm"

key-files:
  created:
    - src/metamemory/transcription/confidence.py
    - src/metamemory/hardware/__init__.py
    - src/metamemory/hardware/detector.py
    - src/metamemory/hardware/recommender.py
    - tests/test_confidence.py
    - tests/test_hardware.py
  modified:
    - src/metamemory/transcription/__init__.py
    - requirements.txt (already had psutil)

key-decisions:
  - "Confidence range: [-3.0,-1.0] log_prob maps to [30,95] score (caps prevent extreme values)"
  - "Distortion effect: Linear from 85% down to 0%, capped at 0.7 for readability"
  - "Model recommendation: <6GB RAM or <4 cores → tiny, <12GB or <8 cores → base, else small"
  - "Hardware cache: 60-second TTL to avoid repeated psutil calls"

patterns-established:
  - "ConfidenceLevel enum with color mapping for consistent UI treatment"
  - "ModelInfo dataclass for model metadata (RAM, accuracy, latency)"
  - "Smart defaults: User override model respected over recommendation"

# Metrics
duration: 5min
completed: 2026-02-02
---

# Phase 2 Plan 3: Confidence Scoring & Hardware Detection Summary

**Confidence normalization from Whisper log_probs with color-coded levels (green/yellow/orange/red), visual distortion effects for low confidence text, psutil-based hardware detection, and intelligent model size recommendations (tiny/base/small) with config persistence**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-02T01:33:08Z
- **Completed:** 2026-02-02T01:38:22Z
- **Tasks:** 5
- **Files modified:** 7

## Accomplishments

- Confidence scoring with normalize_confidence(): Maps Whisper log_probs [-3.0,-1.0] to scores [30,95]
- Four-tier confidence levels: HIGH (80-100%/green), MEDIUM (70-80%/yellow), LOW (50-70%/orange), VERY_LOW (0-50%/red)
- Visual distortion intensity calculation: 0.0 at 85% confidence, linear increase to 0.7 at 0% (capped for readability)
- Hardware detection via psutil: RAM, CPU count (logical/physical), frequency, platform, 64-bit detection
- 60-second result caching with refresh() method for hardware detection
- Model recommendation algorithm from RESEARCH.md: RAM<6GB or CPU<4 → tiny, RAM<12GB or CPU<8 → base, else small
- ConfigManager integration: saves recommendations to hardware.recommended_model, detected specs to last_detected_* fields
- User override support: hardware.user_override_model respected over automatic recommendation
- 51 tests covering confidence normalization edge cases, color coding, distortion calculation, hardware detection, model recommendations

## Task Commits

1. **Task 1: Confidence normalization and color coding** - `88d4be6` (feat)
2. **Task 2: Hardware detection with psutil** - `6e6d54d` (feat)
3. **Task 3: Model recommendation engine** - `161ac2d` (feat)
4. **Task 4: Confidence legend and UI helpers** - (included in Task 1)
5. **Task 5: Integration tests** - `68c3718` (test)
6. **Export confidence from transcription module** - `d0d681e` (chore)

**Plan metadata:** [to be committed]

## Files Created/Modified

- `src/metamemory/transcription/confidence.py` - Confidence normalization, color coding, distortion, legend
- `src/metamemory/hardware/__init__.py` - Hardware module exports
- `src/metamemory/hardware/detector.py` - SystemSpecs, HardwareDetector with psutil
- `src/metamemory/hardware/recommender.py` - ModelRecommender, recommendation algorithm
- `tests/test_confidence.py` - 31 tests for confidence module
- `tests/test_hardware.py` - 20 tests for hardware and recommender
- `src/metamemory/transcription/__init__.py` - Export confidence functions

## Decisions Made

- Confidence normalization uses linear mapping with clamps: log_prob > -1.0 → 95, log_prob < -3.0 → 30
- Distortion effect capped at 0.7 to keep text readable even at 0% confidence
- Four confidence levels match UI requirements from PROJECT.md (TRAN-04, TRAN-06)
- Model recommendation thresholds based on RESEARCH.md research (line 418-428)
- Hardware detection caches for 60 seconds to avoid psutil overhead on repeated calls
- Config integration saves both recommendation and detected specs for future reference

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- ✅ Confidence scoring ready for Phase 5 widget (TRAN-04 color coding, TRAN-06 legend)
- ✅ Hardware detection ready for CFG-05 (display capabilities) and CFG-06 (recommendations)
- ✅ Model recommender integrates with ConfigManager from 02-02
- 🔄 Phase 2 Plan 4 (02-04) ready: Integration & UI wiring

**Blockers:** None

---
*Phase: 02-real-time-transcription-engine*
*Completed: 2026-02-02*
