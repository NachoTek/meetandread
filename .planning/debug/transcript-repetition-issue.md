---
status: investigating
trigger: "Transcript file contains repeating/accumulating text. System outputs entire whisper model result each pass instead of only new content."
created: 2026-02-06T00:00:00Z
updated: 2026-02-06T00:00:00Z
symptoms_prefilled: true
goal: find_root_cause_only
---

## Current Focus

hypothesis: The accumulating transcription processor is outputting the entire accumulated buffer's transcription on each update, and all of it gets added to the transcript store instead of only the new/updated parts.
test: Analyze the flow from accumulating_processor to transcript_store
transcript_store
expecting: Find where full text is being added repeatedly instead of incremental updates
next_action: Document the root cause and provide fix recommendation

## Symptoms

expected: Transcript file should contain each word only once, with incremental additions as new speech is transcribed
actual: Transcript contains repeating/accumulating text. Example: "Testing one two. Testing one two testing. Hello, this is a test of Testing one two testing..."
errors: No error messages - logic error causing text duplication
reproduction: Record audio, let transcription run for multiple update cycles (every 2 seconds), observe transcript file
started: Phase 2 UAT

## Eliminated

## Evidence

- timestamp: 2026-02-06
  checked: accumulating_processor.py _transcribe_accumulated method
  found: Method transcribes entire _phrase_bytes buffer each time (lines 314-388)
  implication: Each transcription returns FULL text for entire buffer, not just new words

- timestamp: 2026-02-06
  checked: accumulating_processor.py _processing_loop
  found: Buffer _phrase_bytes is only cleared when phrase_complete=True (silence detected, lines 293-301)
  implication: Between silence periods, buffer accumulates and gets re-transcribed repeatedly

- timestamp: 2026-02-06
  checked: controller.py _on_phrase_result and _segment_to_words
  found: All segments from transcription are converted to words and added to transcript_store (lines 380-435)
  implication: Every transcription cycle adds ALL words from the buffer, causing duplication

- timestamp: 2026-02-06
  checked: transcript_store.py add_words method
  found: Simply extends _words list with all provided words, no deduplication (lines 124-135)
  implication: Transcript store accumulates all words passed to it, including duplicates

## Resolution

root_cause: In accumulating_processor.py, the _transcribe_accumulated method outputs ALL segments from the transcription on every update cycle (every 2 seconds), not just new/changed segments. Since the audio buffer accumulates until silence is detected (3 seconds), each transcription returns the full accumulated text. The controller's _on_phrase_result adds all these words to the transcript store every cycle, causing exponential duplication.

fix: Need to implement deduplication or incremental update logic
verification: Not yet verified
files_changed: []
