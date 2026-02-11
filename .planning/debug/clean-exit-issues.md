---
status: resolved
trigger: "Debug the clean exit issues from Phase 2 UAT - right-click menu inaccessible, ALT+F4 closes widget but app continues running, CTRL+C required, transcript panel missing close button/lobe"
created: 2026-02-06T00:00:00Z
updated: 2026-02-06T00:00:00Z
---

## Current Focus

hypothesis: Root causes identified for all four issues. Need to document and provide fix recommendations.
next_action: Return structured root cause analysis

## Symptoms

expected: 
1. Right-click on widget shows context menu
2. ALT+F4 closes widget AND exits application cleanly
3. Transcript panel has close button or lobe for dismissal

actual:
1. Right-click menu on widget is inaccessible
2. ALT+F4 closes widget but app continues running
3. Must use CTRL+C which produces KeyboardInterrupt error
4. Transcript panel has no close button or lobe as intended

reproduction:
1. Start the application
2. Try right-clicking on the main widget
3. Try ALT+F4 to close
4. Observe transcript panel lack of close controls

## Eliminated

## Evidence

- timestamp: 2026-02-06
  checked: src/metamemory/widgets/main_widget.py
  found: MeetAndReadWidget has no contextMenuPolicy set, no customContextMenuRequested connection, no QMenu defined
  implication: Issue #1 - Right-click menu not implemented

- timestamp: 2026-02-06
  checked: src/metamemory/widgets/main_widget.py
  found: MeetAndReadWidget has no closeEvent override, no keyPressEvent for ALT+F4, window has Qt.WindowType.Tool flag
  implication: Issue #2 - ALT+F4 doesn't trigger application quit because widget has no close handling and uses Tool window type

- timestamp: 2026-02-06
  checked: src/metamemory/widgets/floating_panels.py
  found: FloatingTranscriptPanel has no close button or lobe in its UI layout
  implication: Issue #4 - No close mechanism exists for transcript panel (unlike settings panel which has a close button)

- timestamp: 2026-02-06
  checked: src/metamemory/main.py
  found: Application uses sys.exit(app.exec()) but has no signal handling for SIGINT/SIGTERM, no aboutToQuit handlers
  implication: CTRL+C causes KeyboardInterrupt because application has no graceful shutdown handling

## Resolution

root_cause: 
1. **No Context Menu Implementation**: MeetAndReadWidget lacks contextMenuPolicy, QMenu, and event handling for right-click
2. **No Close Event Handling**: MeetAndReadWidget doesn't override closeEvent() to trigger application quit; WindowType.Tool prevents ALT+F4 from working properly
3. **No Application Quit Connection**: Widget close doesn't signal QApplication to quit
4. **Missing Close UI for Transcript Panel**: FloatingTranscriptPanel lacks close button/lobe unlike FloatingSettingsPanel which has one

fix: See root cause explanation and fix recommendations below
verification: Pending implementation
files_changed: []
