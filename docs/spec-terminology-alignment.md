# Terminology alignment: scrub → re-transcribe, History → Library, capture nouns → Recording

## Problem Statement

The codebase, README, and UI use terminology that conflicts with the project's domain glossary (CONTEXT.md). A terminology audit found 26 violations across README.md and floating_panels.py. The grilling session refined this to a smaller set of real violations and one glossary correction: the glossary itself had "Scrub" as canonical when "Re-transcribe" is the natural, unambiguous, and discoverable term.

The three categories of violations are:

1. **"Scrub" used where "Re-transcribe" should be** — the glossary has been corrected, but 337 source references, 7 test files, a CSS action property, a module filename, a class name, and a sidecar file-naming pattern all use "scrub".
2. **"History" used where "Library" should be** — the UI tab label and nav button show "History"; the glossary defines this as "Library".
3. **"capture" used as a noun** and **"Audio Input" used as a label** where the glossary defines "Recording" and "Audio Source".

## Solution

A systematic rename across prose, user-facing strings, internal identifiers, and file names to align the codebase with the domain glossary.

## User Stories

### Glossary correction (already done)

1. As a contributor, I want the glossary to list "Re-transcribe" as canonical and "Scrub" in the avoid list, so that I know which term to use when writing new code or documentation.

### scrub → re-transcribe (source code)

2. As a contributor, I want the `ScrubRunner` class renamed to `RetranscribeRunner`, so that the domain term in code matches the glossary.
3. As a contributor, I want the `scrub.py` module renamed to `retranscribe.py`, so that the module name matches the class it contains.
4. As a contributor, I want all `_scrub_`-prefixed variables, methods, and attributes in `floating_panels.py` renamed to `_retranscribe_`-prefixed equivalents, so that internal naming is consistent.
5. As a contributor, I want the CSS `action="scrub"` property renamed to `action="retranscribe"` in theme.py, so that theme styling keys match the domain term.
6. As a contributor, I want `transcript_scanner.py` updated to skip files matching `_retranscribe_` instead of `_scrub_`, so that sidecar files are correctly excluded from the Library listing.
7. As a contributor, I want all user-facing strings (tooltips, dialog titles, button labels, error messages) that say "re-transcribe" to remain unchanged (they are already correct after the glossary flip), and any that say "scrub" to be updated.
8. As a user with existing sidecar files named `{stem}_scrub_{model}.md`, I want the system to still recognize and handle them, so that my in-progress re-transcribe comparisons are not orphaned.
9. As a contributor, I want new sidecar files named `{stem}_retranscribe_{model}.md`, so that the file-naming convention matches the domain term.
10. As a contributor, I want the 7 test files updated to reference the new class name, module path, and file-naming pattern, so that the test suite passes.

### History → Library (UI)

11. As a user, I want the Settings panel nav button to say "Library" instead of "History", so that the term matches the product's domain language.
12. As a user, I want the transcript list tab label to say "Library" instead of "History", so that the term matches the product's domain language.
13. As a user reading the README, I want references to the "History tab" to say "Library tab", so that the documentation matches what I see in the UI.

### capture (noun) → Recording, Audio Input → Audio Source

14. As a user reading the README, I want the heading "Audio Capture" changed to a phrase using "Recording", so that the documentation matches the domain glossary.
15. As a user reading the README, I want the pipeline diagram to label the audio input stage as "Audio Source" instead of "Audio Input", so that the diagram uses canonical domain terms.
16. As a contributor reading comments in `floating_panels.py`, I want noun usages of "capture" (e.g. "audio capture" as a thing, not a verb) rephrased, so that comments use canonical domain terms.

### Follow-up

17. As a contributor, I want the `capture/` source directory renamed to match the domain term, so that the project structure reflects the domain language. (Separate task — requires import changes across the codebase.)

## Implementation Decisions

- **ScrubRunner → RetranscribeRunner**: Full rename of the class in `src/meetandread/transcription/scrub.py` (which becomes `retranscribe.py`). All method names (`scrub_recording`, `accept_scrub`, `reject_scrub`, `sidecar_path`) stay the same internally or get renamed to `retranscribe_recording`, `accept_retranscribe`, `reject_retranscribe`, `sidecar_path`. Constructor signature unchanged.

- **Sidecar file-naming migration**: `sidecar_path()` changes to produce `{stem}_retranscribe_{model}.md`. `transcript_scanner.py` is updated to skip both `_scrub_` and `_retranscribe_` patterns (backward compat). No active migration of existing files — they are transient artifacts (deleted after accept/reject) and will naturally disappear.

- **floating_panels.py dual-panel scrub code**: Both `FloatingTranscriptPanel` (legacy) and `FloatingSettingsPanel` (aetheric) have duplicated re-transcribe UI logic. All `_scrub_`-prefixed identifiers in both panels are renamed to `_retranscribe_`. The `_HistoryRowWidget` inline button follows the same pattern.

- **CSS action property**: `action="scrub"` in theme.py becomes `action="retranscribe"`. CSS selectors and `setProperty` calls are updated to match.

- **"History" scope**: Only user-visible strings are changed (tab label, nav button label). Internal identifiers like `_history_list`, `_refresh_history`, `_HistoryRowWidget` are NOT renamed — that would be a separate, larger refactor.

- **"capture" scope**: Only noun usages are changed. Verb usages ("captures audio", "as it's captured") are left as general English.

- **"archive" in "archive quality"**: Left unchanged — it is a general-English adjective, not a domain term competing with Library.

- **"input" in prose**: Left unchanged except the ASCII diagram label. "Microphone input" is general English for the incoming signal.

## Testing Decisions

- **What makes a good test**: Tests should verify external behavior (file naming, accept/reject outcomes, signal chains, UI state after operations) — not internal variable names. The rename is mechanical; the test concern is that no behavioral regressions are introduced.

- **Primary seam**: `RetranscribeRunner` (formerly `ScrubRunner`). Tests patch `_get_or_create_engine` to avoid loading real models, use `tmp_path` for file system artifacts, and inject callbacks via the constructor. This seam is unchanged by the rename — only the import path and class name change.

- **Secondary seam**: Module-level patching (`"meetandread.transcription.scrub.ScrubRunner"`) used by `test_settings_history.py` — the import path changes to `"meetandread.transcription.retranscribe.RetranscribeRunner"`.

- **Modules tested**:
  - `test_scrub.py` → `test_retranscribe.py`: All 8 test classes updated. `TestSidecarNaming` asserts the new `_retranscribe_` pattern; a new test or expanded existing test verifies backward-compat scanning of `_scrub_` files.
  - `test_settings_history.py`: Import paths and class references updated.
  - `test_settings_history_playback.py`: Already skipped; import references updated.
  - `test_recording_management.py`: Sidecar naming assertions updated.
  - `test_audio_utils.py`, `test_code_deduplication_regressions.py`: Import path updated.
  - `test_theme.py`, `test_aetheric_settings_shell.py`: CSS `action="retranscribe"` assertions updated.

- **Prior art**: Existing tests demonstrate the exact seam patterns needed. No new test infrastructure is required.

## Out of Scope

- **Renaming internal `history`-prefixed identifiers** (`_history_list`, `_refresh_history`, `_HistoryRowWidget`, `HistoryPlaybackController`, etc.). These are a separate, larger refactor.
- **Renaming the `capture/` source directory**. This affects imports across the codebase and should be a separate task.
- **Renaming `benchmark_history`** config key or references. "Benchmark history" refers to a log of benchmark results, not the Library of recordings — it is a different concept.
- **UI layout or behavioral changes**. This is purely a terminology rename.

## Further Notes

- The glossary entry in CONTEXT.md has already been updated (Scrub → Re-transcribe, avoid list flipped).
- The audit's original count of 26 violations was refined through grilling to approximately 9 prose/UI fixes plus the full scrub→retranscribe identifier rename.
- The `gh` CLI is not authenticated, so this spec could not be published as a GitHub issue. Apply the `ready-for-agent` label when creating it manually.