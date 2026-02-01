---
status: investigating
trigger: "Gap: Widget can be dragged from empty/non-interactive areas; user reports empty areas click-through to apps below; expected drag surface captures input. Repo: src/metamemory/widgets/main_widget.py."
created: 2026-02-01T17:13:45-05:00
updated: 2026-02-01T17:22:03-05:00
---

## Current Focus

hypothesis: The widget is a translucent, frameless top-level view with no hit-testable background surface; only QGraphicsItems (buttons/lobes) accept mouse events, and empty pixels are effectively treated as transparent to input on Windows. Separately, drag is broken because is_dragging is never set True after the click-vs-drag threshold refactor.
test: Confirm lack of background item + missing drag state transitions directly in main_widget.py; search codebase for explicit mouse/input transparency flags to rule out intentional click-through.
expecting: main_widget.py shows no background rect/path covering the scene, and no code path sets is_dragging=True; repo search finds no WA_TransparentForMouseEvents/WindowTransparentForInput usage.
next_action: Document specific missing transitions and propose minimal background drag-surface item approach that preserves interactive item clicks.

## Symptoms

expected: User can drag the widget by grabbing a non-interactive surface; clicking empty areas should not click-through to underlying apps.
actual: User reports there is no "empty" area; everything not a button is empty space that clicks through to applications below.
errors: None reported.
reproduction: Hover/click/drag on non-button surface of widget; observe whether underlying app receives clicks and whether widget moves.
started: After recent click-vs-drag threshold changes (suspected).

## Eliminated

- hypothesis: Click-through is caused by explicit Qt WA_TransparentForMouseEvents / WindowTransparentForInput or Win32 WS_EX_TRANSPARENT being set somewhere else in src.
  evidence: Repo search found no references to WA_TransparentForMouseEvents, WindowTransparentForInput, WM_NCHITTEST/HTTRANSPARENT, WS_EX_TRANSPARENT, SetWindowLong/GetWindowLong.
  timestamp: 2026-02-01T17:18:52-05:00

## Evidence

- timestamp: 2026-02-01T17:16:30-05:00
  checked: src/metamemory/widgets/main_widget.py (MeetAndReadWidget init + window flags)
  found: Top-level QGraphicsView is frameless + always-on-top + tool; WA_TranslucentBackground and WA_NoSystemBackground; stylesheet sets fully transparent background.
  implication: Visually transparent areas exist; without an explicit background surface, the only painted pixels are from scene items.

- timestamp: 2026-02-01T17:17:10-05:00
  checked: src/metamemory/widgets/main_widget.py (scene contents)
  found: Scene contains only button/lobe items and an error indicator; no full-rect background item that can accept mouse events.
  implication: "Empty" areas have no scene item to hit-test against; behavior depends on Qt/OS hit testing for translucent windows and can present as click-through.

- timestamp: 2026-02-01T17:18:05-05:00
  checked: src/metamemory/widgets/main_widget.py (mousePressEvent/mouseMoveEvent/mouseReleaseEvent)
  found: is_dragging is initialized False and never set True; mouseMoveEvent only moves the window when is_dragging is already True; release decides click vs drag but does not initiate dragging.
  implication: Dragging the widget cannot start with current code, regardless of thresholds.

## Resolution

root_cause: "MeetAndReadWidget is a frameless translucent top-level QGraphicsView with a fully transparent background and no full-rect background item; only the button/lobe QGraphicsItems are hit-testable/accept mouse, so non-button pixels behave as input-transparent (click-through). Additionally, dragging cannot start because is_dragging is never set True after the click-vs-drag threshold refactor (mouseMoveEvent only moves when already dragging)."
fix: "Add an invisible-but-hit-testable background/drag-surface QGraphicsItem behind all controls (paint with ~alpha=1 and accept left mouse), and implement drag start in mouseMoveEvent when movement exceeds threshold (set is_dragging True, move window while dragging, accept events; release ends drag and snaps)."
verification: "Click empty widget areas: underlying apps must not receive clicks; drag from empty areas moves widget; clicks on record/toggle/settings items still trigger their actions; snap-to-edge still works after dragging."
files_changed: []
