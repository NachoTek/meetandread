---
status: investigating
trigger: "auto-scroll pause issue - user scrolls up but immediately fights to scroll back down instead of pausing auto-scroll for ~10 seconds"
created: "2026-02-06T00:00:00Z"
updated: "2026-02-06T00:00:00Z"
---

## Current Focus

hypothesis: "The auto-scroll mechanism lacks user scroll detection and pause timer implementation"
test: "Analyze FloatingTranscriptPanel scroll handling code"
expecting: "Find where user scroll should be detected and where pause timer should be implemented"
next_action: "Complete analysis and document findings"

## Symptoms

expected: "When user scrolls up manually in transcript panel, auto-scroll should pause for ~10 seconds, allowing reading"
actual: "Auto-scroll immediately fights to scroll back down instead of pausing, making it impossible to read previous content"
errors: "No error messages - behavior issue"
reproduction: "1. Start recording to populate transcript panel\n2. Scroll up manually in the transcript\n3. Observe that it immediately scrolls back down"
started: "Always present in current implementation"

## Eliminated

- None yet

## Evidence

- timestamp: "2026-02-06T00:00:00Z"
  checked: "FloatingTranscriptPanel.__init__() - lines 39-116"
  found: "Auto-scroll timer created but NO user scroll detection: self.scroll_timer = QTimer(self), self.scroll_timer.timeout.connect(self._scroll_to_bottom)"
  implication: "Timer unconditionally calls _scroll_to_bottom every 100ms with no regard for user interaction"

- timestamp: "2026-02-06T00:00:00Z"
  checked: "FloatingTranscriptPanel._scroll_to_bottom() - lines 262-265"
  found: "Method unconditionally sets scrollbar to maximum: scrollbar.setValue(scrollbar.maximum())"
  implication: "No check for whether user manually scrolled up - always forces to bottom"

- timestamp: "2026-02-06T00:00:00Z"
  checked: "FloatingTranscriptPanel.show_panel() - lines 146-152"
  found: "self.scroll_timer.start(100) starts auto-scroll timer running every 100ms"
  implication: "Timer runs continuously without any pause mechanism"

- timestamp: "2026-02-06T00:00:00Z"
  checked: "Entire FloatingTranscriptPanel class for scroll event handling"
  found: "NO scroll event listeners or handlers connected to self.text_edit.verticalScrollBar()"
  implication: "Application has no way to detect when user manually scrolls"

- timestamp: "2026-02-06T00:00:00Z"
  checked: "update_segment() method - lines 172-220"
  found: "Directly calls self._scroll_to_bottom() after each update AND timer also calls it every 100ms"
  implication: "Double auto-scroll: both timer-driven and event-driven, neither respects user scroll"

## Resolution

root_cause: "Complete absence of user scroll detection and pause mechanism. The FloatingTranscriptPanel has a scroll timer that unconditionally forces scrolling to bottom every 100ms without checking if user manually scrolled up. Additionally, update_segment() directly calls _scroll_to_bottom() after every text update."

fix: ""
verification: ""
files_changed: []
