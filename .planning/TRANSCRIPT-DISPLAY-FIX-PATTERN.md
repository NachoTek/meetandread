# Transcript Display Fix Pattern

**Issue Date:** 2026-02-11
**Related Files:**
- `src/metamemory/widgets/main_widget.py`
- `src/metamemory/widgets/floating_panels.py`

## Problem

The transcript display had three recurring issues:

1. **Enhancement behavior not working** - Enhanced segments were not being displayed in bold
2. **Overwriting/flicker effect** - The display would clear and rebuild every time a segment was updated
3. **Confidence color coding unclear** - It wasn't clear if the entire text was confident or just specific segments

## Root Causes

### 1. Missing `enhanced` Parameter in Signal Chain

**Problem:** The `_on_panel_segment` method in `main_widget.py` was missing the `enhanced` parameter.

**Signal Chain:**
```
streaming_pipeline.py: _on_phrase_result() 
  -> emits segment_ready with 6 params (text, confidence, segment_index, is_final, phrase_start, enhanced)
main_widget.py: _on_panel_segment()
  -> RECEIVED only 5 params (missing enhanced!)
  -> calls panel.update_segment with only 5 params
floating_panels.py: update_segment()
  -> enhanced parameter default=False, never gets True
```

**Result:** The `is_enhanced` flag was lost in the signal chain, so bold formatting never triggered.

### 2. Full Display Rebuild Causing Flicker

**Problem:** The `_rebuild_display()` method called `self.text_edit.clear()` and rebuilt the entire transcript from scratch on every segment update.

**Old Implementation:**
```python
def _rebuild_display(self) -> None:
    self.text_edit.clear()  # <-- Clears entire display
    cursor = self.text_edit.textCursor()
    for phrase_idx, phrase in enumerate(self.phrases):
        for seg_idx, (text, conf) in enumerate(zip(phrase.segments, phrase.confidences)):
            # ... rebuild everything
```

**Result:** Every segment update caused the entire transcript to disappear and reappear, creating a flicker/overwrite effect. After a pause, the user would see all previous text vanish and re-render.

### 3. Bold Formatting Based on Confidence

**Problem:** The original implementation had `if conf >= 80 or is_enhanced` for bold formatting.

**Old Code:**
```python
fmt.setFontWeight(QFont.Weight.Bold if conf >= 80 or is_enhanced else QFont.Weight.Normal)
```

**Result:** High-confidence segments appeared bold even when not enhanced, confusing the visual distinction between real-time and enhanced text.

## Solution

### 1. Fix Signal Chain - Add `enhanced` Parameter

**File:** `src/metamemory/widgets/main_widget.py`

**Change:**
```python
# BEFORE:
def _on_panel_segment(self, text: str, confidence: int, segment_index: int, is_final: bool, phrase_start: bool):

# AFTER:
def _on_panel_segment(self, text: str, confidence: int, segment_index: int, is_final: bool, phrase_start: bool, enhanced: bool = False):
    """Handle segment signal from panel (runs on main thread)."""
    print(f"DEBUG Panel Signal: text='{text[:30]}...', idx={segment_index}, phrase_start={phrase_start}, enhanced={enhanced}")
    try:
        self._floating_transcript_panel.update_segment(
            text=text,
            confidence=confidence,
            segment_index=segment_index,
            is_final=is_final,
            phrase_start=phrase_start,
            enhanced=enhanced  # <-- Pass enhanced flag
        )
```

### 2. Implement Incremental Updates Instead of Full Rebuild

**File:** `src/metamemory/widgets/floating_panels.py`

**Change:** Removed `_rebuild_display()` method and replaced with incremental update methods:

**Key Methods:**

1. **`_append_segment_to_display()`** - Adds new segment to end of current line
```python
def _append_segment_to_display(self, text: str, confidence: int, enhanced: bool) -> None:
    """Append a segment to the current line with proper formatting."""
    cursor = self.text_edit.textCursor()
    cursor.movePosition(QTextCursor.MoveOperation.End)
    
    # Add space between segments
    if self.phrases[self.current_phrase_idx].segments and len(self.phrases[self.current_phrase_idx].segments) > 0:
        cursor.insertText(" ")
    
    # Determine color and formatting
    color = self._get_confidence_color(confidence)
    fmt = QTextCharFormat()
    fmt.setForeground(QColor(color))
    # ONLY bold if enhanced, not based on confidence
    fmt.setFontWeight(QFont.Weight.Bold if enhanced else QFont.Weight.Normal)
    
    cursor.insertText(text, fmt)
```

2. **`_replace_segment_in_display()`** - Updates specific segment in place without clearing display
```python
def _replace_segment_in_display(self, phrase_idx: int, segment_idx: int, text: str, confidence: int, enhanced: bool) -> None:
    """Replace a specific segment in the display without rebuilding everything."""
    cursor = self.text_edit.textCursor()
    
    # Move to start of document
    cursor.movePosition(QTextCursor.MoveOperation.Start)
    
    # Navigate to correct phrase block
    for _ in range(phrase_idx):
        cursor.movePosition(QTextCursor.MoveOperation.NextBlock)
    
    # Navigate to correct segment within phrase
    for _ in range(segment_idx):
        cursor.movePosition(QTextCursor.MoveOperation.NextWord)
        # Skip space between segments
        if _ < segment_idx - 1:
            cursor.movePosition(QTextCursor.MoveOperation.NextCharacter)
    
    # Select the segment text
    cursor.movePosition(QTextCursor.MoveOperation.StartOfWord)
    if segment_idx < len(self.phrases[phrase_idx].segments) - 1:
        cursor.movePosition(QTextCursor.MoveOperation.EndOfWord, QTextCursor.MoveMode.KeepAnchor)
    else:
        cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
    
    # Replace with new text and formatting
    color = self._get_confidence_color(confidence)
    fmt = QTextCharFormat()
    fmt.setForeground(QColor(color))
    fmt.setFontWeight(QFont.Weight.Bold if enhanced else QFont.Weight.Normal)
    cursor.insertText(text, fmt)
```

**Modified `update_segment()` logic:**
```python
# Update or add segment
if segment_index < len(phrase.segments):
    # Update existing segment - REPLACE IN PLACE
    phrase.segments[segment_index] = text
    phrase.confidences[segment_index] = confidence
    phrase.enhanced[segment_index] = enhanced
    # Find and replace just this segment in display
    self._replace_segment_in_display(self.current_phrase_idx, segment_index, text, confidence, enhanced)
else:
    # Add new segment - APPEND TO CURRENT LINE
    phrase.segments.append(text)
    phrase.confidences.append(confidence)
    phrase.enhanced.append(enhanced)
    # Append segment to display with proper formatting
    self._append_segment_to_display(text, confidence, enhanced)
```

### 3. Fix Bold Formatting Logic

**Change:** Remove confidence-based bold, only bold enhanced segments

```python
# BEFORE:
fmt.setFontWeight(QFont.Weight.Bold if conf >= 80 or is_enhanced else QFont.Weight.Normal)

# AFTER:
fmt.setFontWeight(QFont.Weight.Bold if enhanced else QFont.Weight.Normal)
```

### 4. Update Test Code

**File:** `src/metamemory/widgets/floating_panels.py` (main block)

Added `enhanced=False` parameter to all test calls.

## Commit History

- `fix(03-04): add missing List import to models.py` - Fixed NameError for List type
- `fix(03-04): add enhanced parameter to _on_panel_segment to enable bold formatting` - Fixed signal chain
- `fix(03-04): implement incremental segment updates to prevent display overwriting` - Fixed flicker issue
- `fix(03-04): update test code to include enhanced parameter` - Fixed test code consistency

## Key Principles for Future Work

### When Working with Qt Text Display:

1. **NEVER clear and rebuild** - This causes flicker and poor UX
2. **Use incremental updates** - Append new text or replace specific selections
3. **Navigate with cursor moves** - Use `MovePosition` with `Start` then navigate to target position
4. **Select with KeepAnchor** - Use `MoveMode.KeepAnchor` to select text range
5. **Replace with insertText** - Replace selection with formatted text

### When Working with PyQt6 Signals:

1. **Match signal and slot parameters** - Ensure slot accepts all parameters from signal
2. **Document parameter order** - Add comments showing expected parameter order
3. **Use default values** - Add `= False` defaults for optional boolean parameters
4. **Add debug logging** - Log parameter values at entry points for debugging

### When Implementing Enhancement Display:

1. **Bold only enhanced segments** - Don't bold based on confidence
2. **Maintain confidence colors** - Keep color coding (green/yellow/orange/red) for quality indication
3. **Track enhancement status** - Pass `is_enhanced` flag through entire update chain
4. **Visual distinction is key** - Users need to clearly see what was enhanced vs real-time

## Testing Checklist

When testing transcript display changes:

- [ ] Enhancement flag propagates through entire signal chain
- [ ] Enhanced segments appear in bold
- [ ] Non-enhanced segments appear in normal weight
- [ ] Confidence color coding still works
- [ ] No flicker when segments are updated
- [ ] No overwriting when segments are added
- [ ] New segments appear at end of current line
- [ ] New phrases appear on new lines
- [ ] Test with FakeAudioModule for reproducible scenarios

## References

- PyQt6 QTextCursor documentation for cursor navigation
- Working implementation in commit `2917d97` (deduplication fix)
- Enhancement architecture: `src/metamemory/transcription/enhancement.py`

---
*Last updated: 2026-02-11*
