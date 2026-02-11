# Metamemory Project Issues Log

## Active Issues

### Issue #1: Live Transcript File Content Issues
**Status:** 🔍 Open  
**Discovered:** 2026-02-05  
**Priority:** Medium

#### Description
After recording completion, the non-enhanced version of the transcript file contains issues with the transcribed live text. The live text that appears correctly in the floating transcript panel during recording may have problems when saved to the file.

#### Context
- The enhanced version (post-processed with base model) appears to work correctly
- Issue is specific to the real-time transcription saved during recording
- Floating transcript panel displays text correctly during live recording
- Problem manifests in the saved `.md` transcript file after recording stops

#### Observed Behavior
- Live text displays correctly in UI during recording
- After recording stops, the saved transcript file may have issues
- This affects the non-enhanced/raw version of the transcript

#### Technical Details
- Location: Transcript files saved in `~/Documents/metamemory/`
- File format: Markdown with timestamp
- Related components:
  - `TranscriptStore` class
  - `RecordingController._on_phrase_result()`
  - `RecordingController._segment_to_words()`
  - Transcript save logic in controller

#### Investigation Notes
- Need to review how segments are converted to Word objects and stored
- Check if segment timing/ordering is preserved correctly
- Verify transcript store state management
- Compare real-time segment flow vs saved file content

#### Possible Causes
1. Segment timing misalignment when converting to Word objects
2. Transcript store not preserving all segments correctly
3. Duplicate or missing word entries in storage
4. Race condition between live updates and final save

#### Action Items
- [ ] Review transcript store add_words() logic
- [ ] Compare stored words vs displayed phrases
- [ ] Check save_transcript() method implementation
- [ ] Add debug logging to track word storage vs display
- [ ] Verify segment-to-word conversion accuracy

## Resolved Issues

### Issue #2: Transcription Display Overwriting Previous Phrases
**Status:** ✅ Resolved  
**Discovered:** 2026-02-05  
**Resolved:** 2026-02-05

#### Problem
Transcription display was overwriting previous content instead of accumulating phrases on new lines.

#### Root Cause
When `phrase_start=True` with "[BLANK_AUDIO]", the code returned early BEFORE creating the new phrase structure. Subsequent segments with real text then updated the previous phrase instead of the new one.

#### Solution
Reordered the logic in `update_segment()` to create the phrase structure BEFORE checking for blank audio content.

#### Changes Made
- `src/metamemory/widgets/floating_panels.py`: Fixed phrase creation order in `update_segment()`
- Added PyQt signal threading for thread-safe UI updates
- Fixed segment indexing to properly track phrases

## Issue Template

When adding new issues, use this format:

```markdown
### Issue #[NUMBER]: [Brief Title]
**Status:** 🔍 Open | 🚧 In Progress | ✅ Resolved  
**Discovered:** YYYY-MM-DD  
**Priority:** Critical | High | Medium | Low

#### Description
[Clear description of the problem]

#### Context
[Background information]

#### Observed Behavior
[What happens vs what should happen]

#### Technical Details
[Code locations, file paths, etc.]

#### Investigation Notes
[Findings as you investigate]

#### Possible Causes
[List of potential causes]

#### Action Items
- [ ] Task 1
- [ ] Task 2
```

---
*Last Updated: 2026-02-05*
