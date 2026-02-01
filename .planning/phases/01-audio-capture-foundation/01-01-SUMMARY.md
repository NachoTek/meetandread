---
phase: 01-audio-capture-foundation
plan: 01
subsystem: storage
tags: [audio, pcm, wav, crash-recovery, streaming]

# Dependency graph
requires:
  - phase: project-setup
    provides: Python project structure with src/ layout
provides:
  - Crash-safe streaming PCM writer with JSON sidecar metadata
  - WAV finalization from raw PCM using stdlib wave module
  - Automatic recovery of partial recordings after crashes
  - Recording directory management (~/Documents/metamemory/)
  - Timestamped filename generation (recording-YYYY-MM-DD-HHMMSS)
affects:
  - Phase 1 remaining plans (capture implementation)
  - Phase 2 (transcription needs playable WAV files)
  - Phase 3 (storage UI needs recovered files)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Sidecar metadata: JSON file alongside binary data for recoverability"
    - "Streaming writes: File handle kept open during recording, flushed periodically"
    - "Context managers: PcmPartWriter supports 'with' statement for cleanup"
    - "Pathlib everywhere: Platform-agnostic path handling"

key-files:
  created:
    - src/metamemory/audio/storage/__init__.py - Module exports
    - src/metamemory/audio/storage/paths.py - Directory/naming utilities
    - src/metamemory/audio/storage/pcm_part.py - Streaming PCM writer
    - src/metamemory/audio/storage/wav_finalize.py - PCM to WAV conversion
    - src/metamemory/audio/storage/recovery.py - Crash recovery
  modified: []

key-decisions:
  - "Use stdlib wave module for WAV headers - avoids hand-rolled header bugs"
  - "PCM + JSON sidecar instead of custom format - simpler, debuggable"
  - "Preserve originals by default on recovery - safer for user data"
  - "Flush API exposed to caller - lets capture control durability vs performance"

patterns-established:
  - "Storage module: paths + writer + finalizer + recovery clear separation"
  - "Stem-based naming: All files share a stem, different extensions"
  - "Metadata dataclass: Type-safe, serializable, documented"

# Metrics
duration: 5min
completed: 2026-02-01
---

# Phase 01 Plan 01: Crash-Safe Audio Storage Summary

**Crash-safe streaming PCM writer with JSON sidecar metadata, WAV finalization via stdlib wave module, and automatic recovery of partial recordings after crashes.**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-01T16:13:30Z
- **Completed:** 2026-02-01T16:18:42Z
- **Tasks:** 3
- **Files created:** 5

## Accomplishments

- Recording directory resolution (~/Documents/metamemory/) with automatic creation
- Timestamped filename generation (recording-YYYY-MM-DD-HHMMSS format)
- Streaming PCM writer that appends int16 frames without loading entire audio into memory
- JSON sidecar metadata enables recovery even if process crashes
- WAV finalization using stdlib `wave` module for reliable headers
- Recovery system detects and converts leftover .pcm.part files to playable WAVs
- Backups of originals (.recovered.bak) preserve user data by default

## Task Commits

Each task was committed atomically:

1. **Task 1: Add recordings path + naming utilities** - `14561d3` (feat)
2. **Task 2: Implement crash-safe .pcm.part writer + WAV finalizer + recovery** - `534cefe` (feat)
3. **Task 3: Add automated tests** - Tests committed with 01-02 plan (already in repo)

**Plan metadata:** (aggregated in this summary)

## Files Created

- `src/metamemory/audio/storage/__init__.py` - Public API exports
- `src/metamemory/audio/storage/paths.py` - Directory resolution and filename utilities
- `src/metamemory/audio/storage/pcm_part.py` - PcmPartWriter with streaming writes
- `src/metamemory/audio/storage/wav_finalize.py` - PCM to WAV conversion
- `src/metamemory/audio/storage/recovery.py` - Find and recover partial recordings
- `tests/test_audio_storage.py` - 25 automated tests (path, writer, finalize, recovery, integration)

## Decisions Made

- Used stdlib `wave` module for WAV header generation instead of hand-rolling - lower risk, well-tested
- Chose PCM + JSON sidecar over custom format - human-readable metadata, easier debugging
- Preserve originals by default on recovery with .recovered.bak suffix - safer default
- Exposed flush() API explicitly - allows capture thread to control durability/performance tradeoff
- Platform-agnostic paths via pathlib - no Windows-specific code needed

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Storage layer is complete and tested
- Ready for integration with audio capture (plan 01-02, 01-03, 01-04)
- Next: WASAPI capture implementation that uses PcmPartWriter for streaming to disk

---
*Phase: 01-audio-capture-foundation*
*Completed: 2026-02-01*
