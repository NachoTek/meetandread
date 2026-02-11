---
status: testing
phase: 02-real-time-transcription-engine
source:
  - 02-01-SUMMARY.md
  - 02-02-SUMMARY.md
  - 02-03-SUMMARY.md
  - 02-04-SUMMARY.md
  - 02-05-SUMMARY.md
  - 02-06-SUMMARY.md
  - 02-07-SUMMARY.md
  - 02-09-SUMMARY.md
  - BUGFIX-dedup-silence-SUMMARY.md
started: 2026-02-09T00:00:00Z
updated: 2026-02-09T00:40:00Z
---

## Current Test

number: 9
name: Transcript File Saved Without Repetition
expected: |
  Stop recording. Check the recording directory (shown in console). File `transcript-{timestamp}.md` exists and contains each phrase only once, without repeating or accumulating text.
awaiting: user response

## Tests

### 2. Settings Panel Opens Without Crash
expected: Click the settings lobe (cog icon) on the widget. Settings panel opens showing hardware detection (RAM, CPU cores), model recommendation (tiny/base/small), and model selection buttons. No AttributeError crash.
result: pass

### 3. Model Selection Persistence
expected: Change model size in settings (e.g., click "base" instead of current selection). Close and restart application. Reopen settings - selected model persists as your last choice.
result: issue
reported: "Exiting application produced errors: 'save_config() takes 0 positional arguments but 1 was given'. The model setting change was not persisted."
severity: major

### 4. Transcription Starts Within 2 Seconds
expected: Click record button. Start speaking clearly. Transcript text appears in the floating panel within 2 seconds of speech beginning.
result: pass

### 5. Confidence Color Coding
expected: As transcribed words appear, they show color-coded confidence - green (high 85-100%), yellow (medium 70-84%), orange (low 50-69%), red (very low 0-49%).
result: pass

### 6. No Duplicate Lines After Silence
expected: Speak a phrase, pause for 3+ seconds (silence detection triggers), speak another phrase. New speech appears on a new line without duplicating the previous phrase.
result: issue
reported: "All output is on one line - silence detection not creating new lines"
severity: major

### 7. Continuous Transcription Without Lag
expected: Record continuously for 2-3 minutes while speaking normally. Transcription keeps pace with speech without accumulating delay or getting progressively slower.
result: issue
reported: "Early in the recording the transcript keeps up fine but as the recording grows in length it slows down and starts getting behind."
severity: major

### 8. Transcript Auto-Scroll Pause
expected: While recording, scroll up manually in the transcript panel to read earlier content. The panel stays at your scroll position (does NOT immediately jump back down). After ~10 seconds of no manual scrolling, auto-scroll resumes and jumps to bottom.
result: issue
reported: "The panel stays where you move it while recording and transcribing but it never jumps back down after a time out - you have to manually move it back to the bottom"
severity: major

### 9. Transcript File Saved Without Repetition
expected: Stop recording. Check the recording directory (shown in console). File `transcript-{timestamp}.md` exists and contains each phrase only once, without repeating or accumulating text.
result: pending

### 10. Widget Dock and Position Persistence
expected: Drag widget to screen edge - it docks showing 4/5ths (translucent). Move to a new position. Close application cleanly (right-click → Exit or ALT+F4). Restart - widget returns to the position you left it.
result: pending

### 11. Clean Exit via Context Menu
expected: Right-click anywhere on the widget. Context menu appears with an "Exit" option. Click "Exit" - application closes cleanly with no error traceback or crash.
result: pending

### 12. Clean Exit via ALT+F4
expected: With widget focused, press ALT+F4. Application closes cleanly with no error traceback or crash.
result: pending

### 13. Clean Exit via CTRL+C
expected: In the terminal where app is running, press CTRL+C. Application shuts down gracefully with a clean shutdown message, no KeyboardInterrupt traceback.
result: pending

## Summary

total: 13
passed: 4
issues: 4
pending: 5
skipped: 0

## Gaps

- truth: "Model selection persists across application restarts"
  status: failed
  reason: "User reported: Exiting application produced errors: 'save_config() takes 0 positional arguments but 1 was given'. The model setting change was not persisted."
  severity: major
  test: 3
  root_cause: ""
  artifacts: []
  missing: []
  debug_session: ""

- truth: "Silence detection creates new lines in transcript"
  status: failed
  reason: "User reported: All output is on one line - silence detection not creating new lines"
  severity: major
  test: 6
  root_cause: ""
  artifacts: []
  missing: []
  debug_session: ""

- truth: "Transcription maintains consistent latency over time"
  status: failed
  reason: "User reported: Early in the recording the transcript keeps up fine but as the recording grows in length it slows down and starts getting behind."
  severity: major
  test: 7
  root_cause: ""
  artifacts: []
  missing: []
  debug_session: ""

- truth: "Auto-scroll resumes after 10-second pause timeout"
  status: failed
  reason: "User reported: The panel stays where you move it while recording and transcribing but it never jumps back down after a time out - you have to manually move it back to the bottom"
  severity: major
  test: 8
  root_cause: ""
  artifacts: []
  missing: []
  debug_session: ""
