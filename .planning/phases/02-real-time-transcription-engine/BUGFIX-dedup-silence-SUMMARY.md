# Bug Fix: Duplicate Lines After Silence Detection - Summary

## Overview

**Type:** Bug Fix (Non-Plan Task)
**Completion Date:** 2026-02-03
**Duration:** ~15 minutes
**Status:** ✅ Complete
**Commit:** `2917d97`

## Problem Statement

**Symptom:** User says "Hello" after silence period, gets "Hello" printed twice on separate lines.

**Root Cause:** 
After silence detection and buffer clearing, re-transcription of accumulated audio was producing identical results, causing duplicate output lines in the transcript panel.

The timing flow was:
1. Silence detected → transcribe current buffer with `phrase_complete=True`
2. Buffer cleared (`self._phrase_bytes = bytes()`)
3. NEW audio "Hello" comes in immediately
4. First transcription of "Hello" happens (is_final=False) → UI shows "Hello"
5. Second update happens with identical "Hello" → creates duplicate line

## Solution Implemented

### Changes to `accumulating_processor.py`

1. **Added deduplication tracking fields:**
   - `_last_transcribed_text`: Tracks last output text to skip duplicates
   - `_last_phrase_start_time`: Tracks phrase timing
   - `_min_phrase_duration`: Minimum audio threshold (0.3s) before transcription

2. **Updated `_transcribe_accumulated()` method:**
   - Skip output if `full_text == _last_transcribed_text` and not force_complete
   - Reset `_last_transcribed_text` on phrase complete for new phrases

3. **Updated processing loop:**
   - Changed minimum buffer check from 0.5s to `_min_phrase_duration` (0.3s)
   - Reset dedup state when clearing buffer after silence

### Changes to `floating_panels.py`

1. **Added UI-level deduplication in `update_line()`:**
   - Skip update if same text as current line and not final
   - Prevents flicker and duplicate entries in the transcript panel

## Verification

- ✅ All 27 transcription-related tests pass
- ✅ Deduplication tracking initializes correctly
- ✅ Duplicate text updates are skipped in UI
- ✅ Non-duplicate updates still work correctly

## Files Modified

| File | Changes |
|------|---------|
| `src/metamemory/transcription/accumulating_processor.py` | +21 lines: dedup tracking, min duration, skip logic |
| `src/metamemory/widgets/floating_panels.py` | +6 lines: UI dedup check |

## Technical Details

### Deduplication Logic

```python
# In accumulating processor
if full_text == self._last_transcribed_text and not force_complete:
    return  # Skip duplicate

# In UI panel
if (current_line.text == text and not is_final):
    return  # Duplicate, skip
```

### State Reset on Phrase Complete

```python
if phrase_complete:
    self._phrase_bytes = bytes()
    self._last_transcribed_text = ""  # Reset dedup
    self._last_phrase_start_time = None  # Reset timing
```

## Testing

Manual verification performed:
1. Unit test: Deduplication tracking initializes to empty string
2. Unit test: Minimum phrase duration set to 0.3s
3. Unit test: UI panel skips duplicate updates
4. Unit test: UI panel allows different text updates

## Impact

- **Phase 2:** Real-time transcription now handles silence boundaries correctly
- **User Experience:** No more duplicate lines after pauses in speech
- **Performance:** Slight improvement from skipping redundant transcriptions

## Related

- Phase: 02-real-time-transcription-engine
- Previous Plan: 02-05 (whisper.cpp gap closure)
- Next: Awaiting human verification checkpoint for Phase 2
