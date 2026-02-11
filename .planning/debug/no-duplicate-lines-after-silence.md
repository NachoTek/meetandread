---
status: investigating
trigger: "Diagnose Issue #3 from Phase 2 UAT: No Duplicate Lines After Silence. Reported: 'Text does not appear on a new line. It continues to append to the old text even after a long pause.'"
created: 2026-02-10T00:00:00.000Z
updated: 2026-02-10T00:00:00.000Z
---

## Current Focus

hypothesis: "ROOT CAUSE CONFIRMED: In AccumulatingTranscriptionProcessor._transcribe_accumulated(), line 342 reads `self._new_phrase_started` into local variable `phrase_start`, but line 343 immediately resets `self._new_phrase_started = False`. Line 412 then uses `self._new_phrase_started` (already False) instead of the local `phrase_start` variable. This causes `phrase_start` to ALWAYS be False, so the UI never creates new phrases after silence."
test: "Trace the flow of the phrase_start flag from line 342 to line 412"
expecting: "Will confirm the bug where the local phrase_start variable is ignored in favor of the reset flag"
next_action: "N/A - Diagnosis complete"

## Symptoms

expected: [Text should appear on a new line after a pause in speech]
actual: [Text continues to append to old line instead of creating new line]
errors: [None reported]
reproduction: [Unknown - need to understand trigger conditions]
started: [Phase 2 UAT testing]

## Eliminated

- None yet

## Evidence

- timestamp: 2026-02-10
  checked: "Found transcript display code in FloatingTranscriptPanel (floating_panels.py)"
  found: "update_segment() method at line 213 handles new segments and phrase_start flag to create new lines"
  implication: "UI correctly implements new line creation when phrase_start=True, so bug must be in data flow"

- timestamp: 2026-02-10
  checked: "Found AccumulatingTranscriptionProcessor on_result callback chain"
  found: "Processor -> controller.on_phrase_result -> main_widget._on_phrase_result -> panel.segment_ready -> panel.update_segment()"
  implication: "phrase_start flag flows through entire chain, so bug is in processor"

- timestamp: 2026-02-10
  checked: "Examined AccumulatingTranscriptionProcessor._transcribe_accumulated() lines 342-412"
  found: "Line 342: phrase_start = self._new_phrase_started (saves to local var) → Line 343: self._new_phrase_started = False (immediate reset) → Line 412: phrase_start=(i == 0 and self._new_phrase_started) (uses RESET variable!)"
  implication: "BUG: Local variable 'phrase_start' is ignored; code uses self._new_phrase_started which is already False"

- timestamp: 2026-02-10
  checked: "Checked silence timeout and phrase complete logic in _processing_loop()"
  found: "Lines 300-311 correctly set _new_phrase_started=True after phrase complete, clear buffer, reset timers"
  implication: "Processor logic is correct; the flag is set properly before transcription"

## Resolution

root_cause: "In `AccumulatingTranscriptionProcessor._transcribe_accumulated()` at lines 342-343, the code reads `self._new_phrase_started` into local variable `phrase_start`, but then immediately resets `self._new_phrase_started = False`. Line 412 then uses `self._new_phrase_started` (already False) instead of the local `phrase_start` variable. This causes the `phrase_start` flag to ALWAYS be False, so the FloatingTranscriptPanel never creates new phrases (new lines) after speech silence is detected."

fix: "Change line 412 to use the local `phrase_start` variable instead of `self._new_phrase_started`. Should be: `phrase_start=(i == 0 and phrase_start)`"

verification: "After fix, phrase_start will correctly propagate from the processing loop to the UI, creating new lines after silence detection"

files_changed: ["src/metamemory/transcription/accumulating_processor.py"]
