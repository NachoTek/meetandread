# WIDGET Design Deviations

Intentional deviations from the original WIDGET requirements specification, documented for traceability.

---

## WIDGET-17: Third lobe for transcript access

| Field | Detail |
|---|---|
| **Requirement ID** | WIDGET-17 |
| **Original Spec** | A dedicated third lobe on the widget for opening/closing the transcript panel. |
| **Actual Implementation** | No dedicated transcript lobe exists. The transcript panel auto-shows when recording starts and can be toggled via the settings lobe or context menu. The FloatingTranscriptPanel is a separate QWidget that docks adjacent to the main widget. |
| **Rationale** | UX simplification — the transcript automatically appearing on recording start eliminates a discoverability step. A dedicated lobe would add visual clutter for an action that should be automatic. Users who want manual control can toggle via the existing lobe or context menu. |
| **Date** | 2025-04-27 |

---

## WIDGET-18: Transcript flows from widget

| Field | Detail |
|---|---|
| **Requirement ID** | WIDGET-18 |
| **Original Spec** | Transcript content flows directly out of the widget body, integrated as a continuous visual extension. |
| **Actual Implementation** | A separate `FloatingTranscriptPanel` QWidget is used. It positions itself adjacent to the main widget via `dock_to_widget()` but is architecturally a standalone window — not an integrated flow-out component within the widget's `QGraphicsView` scene. |
| **Rationale** | Pragmatic architecture — integrating a scrolling text panel within the QGraphicsView scene that also hosts orbital animations would create complex layout conflicts and performance issues. A separate top-level QWidget provides clean separation of concerns, independent scrolling, and proper window management while maintaining visual adjacency. |
| **Date** | 2025-04-27 |

---

## WIDGET-29: Enhanced segment bold styling

| Field | Detail |
|---|---|
| **Requirement ID** | WIDGET-29 |
| **Original Spec** | Enhanced bold styling for speaker-change segments in the transcript. |
| **Actual Implementation** | Dual-mode enhancement was removed during M001/S03 cleanup. Speaker segments use standard bold formatting via the transcript panel's `append_speaker_segment()` without a special enhanced mode. |
| **Rationale** | The dual-mode (normal/enhanced) styling added UI complexity without proportional user value. Standard bold speaker labels with colored indicators (WIDGET-21/22/23) provide sufficient visual differentiation. The enhancement was cut during initial cleanup to reduce surface area. |
| **Date** | 2025-04-27 |
