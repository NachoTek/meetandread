# S04-VISUAL-VERIFICATION.md

**Milestone:** M006-mgkqrb — Aetheric Glass Desktop Redesign
**Slice:** S04 — Integrated Aetheric Desktop Regression and Visual Completion
**Task:** T02 — Write Windows desktop visual UAT checklist and completion artifact
**Date:** 2026-04-29T16:22:00 UTC

---

## Automated Regression Baseline

This visual UAT is the second half of S04 verification. The automated regression proof (T01) completed with:

- **Result:** 1037/1038 tests pass (1 pre-existing config singleton failure unrelated to Aetheric Glass)
- **S01 contracts:** Settings shell, Aetheric theme, docking sync — all green (124 tests)
- **S02 contracts:** Settings History migration — all green (62 tests)
- **S03 contracts:** CC overlay lifecycle — all green (71 tests)
- **Details:** See `S04-REGRESSION-RESULTS.md` for full command history and per-surface breakdown.

**Verdict:** Automated regression does NOT block visual UAT. Proceed with desktop verification.

---

## Launch Instructions

### Prerequisites

1. **Environment:** Windows 11 (22H2 or later) with DWM compositor active
2. **Display:** Recommend 1920x1080 or higher; note display scaling (100%, 125%, 150%, etc.)
3. **Test Data:** At least one completed recording in `~/.meetandread/recordings/` for History tab verification
4. **Audio:** Microphone (any) and optional system loopback (if available) for CC overlay test
5. **Application:** Launch `python -m meetandread` from this working tree

### Launch Steps

1. Open a terminal: `cd C:/Users/david.keymel/Documents/Projects/meetandread/.gsd/worktrees/M006-mgkqrb`
2. Run: `python -m meetandread`
3. The widget should appear as a floating glass surface with lobes (mic, system audio, transcript lobe, settings lobe, etc.)

---

## Visual UAT Checklist

### 3.1 Settings Shell

| # | Check | Expected Observation | Owner File on Failure |
|---|-------|---------------------|----------------------|
| 1.1 | **Widget settings lobe opens docked Settings** | Click the settings lobe (gear icon) on the widget; the Aetheric Glass Settings panel opens aligned around the widget position with the widget visually docked into the panel's bottom-left sidebar area | `main_widget.py` + `floating_panels.py` |
| 1.2 | **Context-menu Settings opens docked Settings** | Right-click the widget → select "Settings"; the same docked Settings panel opens (should be identical to lobe-triggered open) | `main_widget.py` + `floating_panels.py` |
| 1.3 | **Frameless chrome, no title bar, no close button** | Settings panel has no window title bar, no close/minimize/maximize buttons. It is a pure glass surface. | `floating_panels.py` |
| 1.4 | **Aetheric Glass styling (translucency, borders, radius)** | Panel background is translucent dark glass (rgba(30,29,30,220) or similar), with directional borders (highlight top-left, shadow bottom-right), 12px corner radius. | `theme.py` |
| 1.5 | **Sidebar nav with Settings/Performance/History** | Left sidebar shows three nav items with icons: Settings (gear), Performance (chart/graph), History (clock/list). Labels use Space Grotesk-style font. | `floating_panels.py` + `theme.py` |
| 1.6 | **Active tab red glow and pill styling** | The active tab (initially Settings) has a red glow/pill background that visually distinguishes it from inactive tabs. | `theme.py` + `floating_panels.py` |
| 1.7 | **Settings tab content renders** | Click Settings nav item; the content area shows the existing Settings controls (model selection, language, audio sources, etc.). All controls are visible and interactive. | `floating_panels.py` (hosting existing settings.py) |
| 1.8 | **Performance tab content renders** | Click Performance nav item; the content area shows live resource monitoring (RAM/CPU bars), recording metrics, benchmark button, auto-WER display. | `floating_panels.py` (hosting performance.py from S05) |
| 1.9 | **History tab shows recording list** | Click History nav item; the content area shows a list of completed recordings with timestamps, durations, and transcript counts. | `floating_panels.py` (hosting history.py from S02) |
| 1.10 | **History transcript view and scrub** | Click a recording in History; the transcript view opens with segment text and timestamps. Scrub through the transcript using the seek bar or arrow keys; playback position updates. | `floating_panels.py` + `history.py` |
| 1.11 | **History delete recording** | Select a recording in History → click Delete (or right-click → Delete); the recording is removed from the list and the file is deleted from disk. Confirm deletion with a dialog if present. | `floating_panels.py` + `history.py` |
| 1.12 | **History speaker rename** | In a transcript view, click a speaker label (e.g., "SPK_0") → enter a name (e.g., "Alice") → confirm. All segments for that speaker update to the new name. | `floating_panels.py` + `history.py` |
| 1.13 | **Settings close undocks widget** | Close the Settings panel (via keyboard Escape or the close action in Settings itself); the panel disappears and the widget returns to its original undocked position. | `floating_panels.py` + `main_widget.py` |

---

### 3.2 CC Overlay

| # | Check | Expected Observation | Owner File on Failure |
|---|-------|---------------------|----------------------|
| 2.1 | **CC overlay fades in on record start** | Click the mic lobe to start recording; a compact CC overlay appears near the widget with a smooth fade-in (150ms). | `floating_panels.py` + `main_widget.py` |
| 2.2 | **No-header chrome, no status text** | CC overlay has no title bar, no status text like "Recording in progress..." — only the live transcript text. Pure display-only surface. | `floating_panels.py` |
| 2.3 | **Translucency and text styling** | Overlay background is `rgba(30,29,30,220)` (dark glass), text is white/light gray with Aetheric typography (Inter-style). Text is readable against the glass. | `theme.py` + `floating_panels.py` |
| 2.4 | **Live transcript updates** | As you speak, transcribed text appears line-by-line in the overlay. No lag, no duplicate lines, text scrolls if it exceeds the visible area. | `floating_panels.py` (wired to RecordingController) |
| 2.5 | **Widget tracking (optional)** | If you move the widget during recording, the CC overlay should follow (or remain anchored relative to widget position) without jitter. | `main_widget.py` + `floating_panels.py` |
| 2.6 | **Manual transcript lobe toggle** | Click the transcript lobe on the widget; the CC overlay should toggle visibility (hide if visible, show if hidden) while recording continues. | `main_widget.py` + `floating_panels.py` |
| 2.7 | **1.5s delayed hide on stop** | Stop recording (click the mic lobe again); the CC overlay remains visible for ~1.5 seconds with final transcript text, then fades out smoothly. No premature dismissal. | `floating_panels.py` |
| 2.8 | **Restart-before-fade cancellation** | Stop recording → wait <1.5s → start recording again before overlay fades out. The overlay should remain visible and resume showing live transcript without flickering or hiding first. | `floating_panels.py` + `main_widget.py` |

---

### 3.3 Drag/Resize/Z-Order

| # | Check | Expected Observation | Owner File on Failure |
|---|-------|---------------------|----------------------|
| 3.1 | **Settings panel is draggable** | Click and drag the Settings panel (anywhere on the glass, not just a header); it moves smoothly with the mouse. The docked widget moves with it. | `floating_panels.py` + `main_widget.py` |
| 3.2 | **Settings panel is resizable** | Grab the edge or corner of the Settings panel; drag to resize. The panel resizes smoothly with the glass and borders adjusting. | `floating_panels.py` |
| 3.3 | **CC overlay is draggable** | Click and drag the CC overlay during recording; it moves independently of the widget and Settings panel. | `floating_panels.py` |
| 3.4 | **CC overlay is resizable** | Grab the edge or corner of the CC overlay; drag to resize. The text area adjusts accordingly. | `floating_panels.py` |
| 3.5 | **Z-order is correct** | Settings panel and CC overlay appear above other windows but below modal dialogs. Overlap between Settings and CC should not cause flickering or incorrect occlusion. | `floating_panels.py` (Qt window flags) |

---

### 3.4 Theme and Animation

| # | Check | Expected Observation | Owner File on Failure |
|---|-------|---------------------|----------------------|
| 4.1 | **DWM translucency is active** | The glass backgrounds on Settings and CC overlay show subtle blur/transparency. You should see desktop content faintly through the panels. | `theme.py` (Qt WA_TranslucentBackground + Qt.FramelessWindowHint) |
| 4.2 | **Directional borders render** | Panel top-left borders have a subtle highlight (lighter edge), bottom-right borders have a subtle shadow (darker edge). Creates depth and "Aetheric" feel. | `theme.py` (border-image or QSS gradients) |
| 4.3 | **Neon active states** | Active tab glow, active settings controls, and hover states use neon accent colors (cyan/teal/red) that stand out against dark glass. | `theme.py` |
| 4.4 | **Dropdown chevrons are V-shaped** | Any dropdown controls (e.g., model selector, language selector) show V-shaped chevron indicators that rotate on open/close (optional animation). | `theme.py` (QSS ::down-arrow) |
| 4.5 | **Typography is clear (Space Grotesk/Inter)** | Tab labels, headings, and body text use Space Grotesk-style headings and Inter-style body text. Font sizes are legible at 100%/125%/150% display scaling. | `theme.py` (font-family QSS) |

---

## Results

### Verifier Information

| Field | Value |
|-------|-------|
| Verifier name | *(enter your name)* |
| Date | *(YYYY-MM-DD)* |
| Windows version | *(e.g., Windows 11 22H2, build 22621)* |
| Display scaling | *(e.g., 150%)* |
| GPU/Compositor | *(e.g., NVIDIA RTX 3060, DWM)* |
| Python version | *(e.g., 3.10.11)* |
| PySide6 version | *(e.g., 6.10.1)* |

### Group Verdicts

| Group | Total Checks | Passed | Failed | Blocked | Verdict |
|-------|-------------|--------|--------|---------|---------|
| 3.1 Settings Shell | 13 | – | – | – | *(PASS / FAIL / NOT RUN)* |
| 3.2 CC Overlay | 8 | – | – | – | *(PASS / FAIL / NOT RUN)* |
| 3.3 Drag/Resize/Z-Order | 5 | – | – | – | *(PASS / FAIL / NOT RUN)* |
| 3.4 Theme/Animation | 5 | – | – | – | *(PASS / FAIL / NOT RUN)* |
| **Overall** | **31** | – | – | – | ***(NOT YET RUN — REQUIRES WINDOWS 11 DESKTOP)*** |

### Per-Check Results

Use the format below for each check that fails or is blocked:

```
[Check ID] - [Short description]
Status: PASS / FAIL / BLOCKED
Observed: [what you saw]
Expected: [what should have happened]
Owner file: [likely source file from table above]
Screenshot: [optional: path to screenshot]
```

---

## Artifact References

- Automated regression: `S04-REGRESSION-RESULTS.md` (1037/1038 tests pass, 1 pre-existing config failure)
- Theme tokens: `src/meetandread/widgets/theme.py` — AethericGlassTheme class
- Settings shell: `src/meetandread/widgets/floating_panels.py` — FloatingSettingsPanel class
- Widget integration: `src/meetandread/widgets/main_widget.py` — MeetAndReadWidget class
- History workflows: `src/meetandread/widgets/history.py` (migrated from S02)
- CC overlay: `src/meetandread/widgets/floating_panels.py` — ClosedCaptionOverlay class

---

## Owner-File Quick Reference

| Symptom | Likely Owner File | Notes |
|---------|------------------|-------|
| Panel won't open / doesn't dock | `main_widget.py` + `floating_panels.py` | Check `_open_settings()` and docking alignment logic |
| Wrong colors / borders / glow | `theme.py` | Check AethericGlassTheme palette and QSS |
| Panel has title bar or close button | `floating_panels.py` | Check window flags (Qt.FramelessWindowHint) |
| Sidebar nav doesn't highlight active tab | `floating_panels.py` + `theme.py` | Check `_update_nav_state()` and active-tab QSS |
| CC overlay doesn't appear on record | `floating_panels.py` + `main_widget.py` | Check `_on_controller_state_change()` wiring |
| CC overlay fades too fast / too slow | `floating_panels.py` | Check `_schedule_hide()` timer value (1500ms) |
| Panel/overlay doesn't drag | `floating_panels.py` | Check mouse event handlers (mousePress, mouseMove) |
| Panel/overlay doesn't resize | `floating_panels.py` | Check size grip or edge detection logic |
| Text is unreadable / wrong font | `theme.py` | Check font-family QSS and color contrast |
| Z-order issues (wrong window on top) | `floating_panels.py` | Check window flags and raise()/lower() calls |

---

## Confidentiality Notice

**Do NOT include transcript content, audio data, or any private meeting information in this artifact or any screenshots attached to it.** The visual UAT should focus on UI behavior, styling, and interaction — not on the actual transcribed words. Redact transcript text before sharing.

---

*This artifact was generated by GSD auto-mode. The actual visual UAT result is "Not yet run — requires Windows 11 desktop" pending human verifier.*
