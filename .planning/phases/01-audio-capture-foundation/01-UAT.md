---
status: diagnosed
phase: 01-audio-capture-foundation
source: 01-05-SUMMARY.md, 01-06-SUMMARY.md, 01-07-SUMMARY.md, 01-08-SUMMARY.md
started: 2026-02-01
updated: 2026-02-01T22:25:00Z
---

## Current Test

[testing complete]

## Tests

### 1. CLI fake recording creates correct duration WAV
expected: |
  Run: python -m metamemory.audio.cli --fake path/to/test.wav --seconds 5
  
  You should see:
  - Recording starts and shows progress
  - Recording stops automatically after ~5 seconds
  - A WAV file is created in the recordings directory
  - The WAV file is approximately 5 seconds long (NOT 3+ hours)
result: issue
reported: "Recording was created and plays but it was the full 9 second length of the test file. a full 1 to 1 copy."
severity: major

### 2. Record button single-click works
expected: |
  Launch widget: python -m meetandread.main
  
  Click the center record button ONCE (not double-click)
  
  You should see:
  - Button changes to recording state (glowing red pulse)
  - Recording starts immediately
  - Click once more to stop - recording stops immediately
result: pass

### 3. Source lobes single-click toggle works
expected: |
  With widget running, click the Mic lobe (left) ONCE
  
  You should see:
  - Lobe toggles between active/inactive (visual color change)
  - Click once more toggles it back
  
  Repeat for System lobe (right)
result: pass

### 4. Click vs drag detection works
expected: |
  With widget running:
  
  1. Click record button and drag slightly - widget should NOT drag, button should trigger
  2. Click on empty widget area and drag - widget SHOULD drag to new position
  3. Click lobe and drag slightly - lobe should toggle, widget should NOT drag
result: issue
reported: "There is no 'empty' area of the widget to try and drag from. everything that is not a button is empty space that clicks through to the applications below it."
severity: major

### 5. Settings lobe single-click works
expected: |
  With widget running, click the Settings lobe (top) ONCE
  
  You should see:
  - Console output appears (settings action triggered)
  - No double-click required
result: pass

### 6. No crash recovery prompt on clean startup
expected: |
  1. Start app, record a short clip (5 seconds), stop cleanly
  2. Check recordings directory - should have .wav file, NO .pcm.part files
  3. Close app completely
  4. Restart app
  
  You should see:
  - App starts normally WITHOUT any recovery prompt
  - No dialog asking about crash leftovers
result: pass

### 7. Crash recovery still works for actual crashes
expected: |
  1. Start recording
  2. Kill the app process during recording (simulate crash)
  3. Restart app
  
  You should see:
  - Recovery dialog appears asking about restoring partial recording
  - Recovery works and restores the partial recording
result: pass

## Summary

total: 7
passed: 5
issues: 2
pending: 0
skipped: 0

## Gaps

- truth: "CLI fake recording creates WAV of specified duration (--seconds N)"
  status: failed
  reason: "User reported: Recording was created and plays but it was the full 9 second length of the test file. a full 1 to 1 copy."
  severity: major
  test: 1
  root_cause: "FakeAudioModule emits WAV blocks as fast as the pipeline can consume them (no real-time pacing), so the full input file can be written before the CLI's wall-clock sleep ends; AudioSession does not cap/truncate output by target duration."
  artifacts:
    - path: "src/metamemory/audio/capture/fake_module.py"
      issue: "_read_loop() emits blocks without pacing, so a 9s file can be fully queued/consumed before stop."
      suggested_fix: "Add real-time pacing per block based on n_frames_read / sample_rate (optionally behind a realtime/speed flag)."
    - path: "src/metamemory/audio/cli.py"
      issue: "--seconds uses time.sleep only; assumes sources are real-time."
      suggested_fix: "Optional: plumb a max_frames/max_duration cap into AudioSession so output duration is enforced regardless of source speed."
    - path: "src/metamemory/audio/session.py"
      issue: "No truncate/cap by target duration; writes whatever frames arrive until stop."
      suggested_fix: "Add optional max_frames guard in the consumer loop to stop writing once enough frames are recorded."
  missing:
    - "Fake source real-time pacing (or explicit max-duration cap) so --seconds produces consistent audio duration"
    - "Optional session-level max_frames guard to enforce requested duration even if a source runs faster than real-time"
    - "Regression test: 9s fake input + --seconds 5 outputs ~5s audio"
  debug_session: ".planning/debug/uat-01-fake-duration.md"
- truth: "Widget can be dragged from empty/non-interactive areas"
  status: failed
  reason: "User reported: There is no 'empty' area of the widget to try and drag from. everything that is not a button is empty space that clicks through to the applications below it."
  severity: major
  test: 4
  root_cause: "The widget's non-button pixels are fully transparent with no full-rect hit-testable surface, so empty areas behave as input-transparent/click-through; additionally, dragging cannot start because is_dragging is never set True after the click-vs-drag refactor (mouseMoveEvent only moves when already dragging)."
  artifacts:
    - path: "src/metamemory/widgets/main_widget.py"
      issue: "Fully transparent background + no full-rect background item; only buttons/lobes are hit-testable, so empty areas can click-through."
      suggested_fix: "Add a near-invisible background/drag-surface QGraphicsItem covering the scene rect (paint alpha ~1) behind controls to capture input and prevent click-through."
    - path: "src/metamemory/widgets/main_widget.py"
      issue: "Drag state machine never enters dragging (is_dragging is never set True)."
      suggested_fix: "Start drag on mouse move when movement exceeds threshold; set is_dragging=True, move window while dragging, end on release and snap."
  missing:
    - "Hit-testable drag surface behind interactive items to capture empty-area input"
    - "Drag-start transition sets is_dragging True once movement exceeds threshold"
    - "Empty-area clicks are consumed by the widget (no click-through)"
  debug_session: ".planning/debug/uat-01-widget-drag-surface.md"
