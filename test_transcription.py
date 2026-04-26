#!/usr/bin/env python3
"""Test script to verify transcription is working."""

import sys
sys.path.insert(0, 'src')

import numpy as np
from meetandread.transcription.accumulating_processor import AccumulatingTranscriptionProcessor, SegmentResult

def test_segment_filtering():
    """Test the segment filtering logic."""
    print("=" * 60)
    print("Testing Segment Filtering Logic")
    print("=" * 60)
    
    # Simulate the processor state
    _last_emitted_segment_index = -1
    
    # First transcription: 1 segment
    print("\n1. First transcription (1 segment):")
    segments_1 = [{'text': 'Hello world', 'confidence': 85, 'start': 0, 'end': 2}]
    new_segments = []
    for i, seg in enumerate(segments_1):
        condition = i > _last_emitted_segment_index
        print(f"   Segment {i}: '{seg['text']}' - Check: {i} > {_last_emitted_segment_index} = {condition}")
        if condition:
            new_segments.append((i, seg))
    
    print(f"   Result: {len(new_segments)} new segments emitted")
    
    # Update tracking
    if segments_1:
        _last_emitted_segment_index = len(segments_1) - 1
        print(f"   Updated _last_emitted_segment_index = {_last_emitted_segment_index}")
    
    # Second transcription: Same segment + 1 new (simulating buffer growth)
    print("\n2. Second transcription (2 segments, 1 new):")
    segments_2 = [
        {'text': 'Hello world', 'confidence': 88, 'start': 0, 'end': 2},
        {'text': 'how are you', 'confidence': 82, 'start': 2, 'end': 4}
    ]
    new_segments = []
    for i, seg in enumerate(segments_2):
        condition = i > _last_emitted_segment_index
        print(f"   Segment {i}: '{seg['text']}' - Check: {i} > {_last_emitted_segment_index} = {condition}")
        if condition:
            new_segments.append((i, seg))
    
    print(f"   Result: {len(new_segments)} new segments emitted")
    
    # Update tracking
    if segments_2:
        _last_emitted_segment_index = len(segments_2) - 1
        print(f"   Updated _last_emitted_segment_index = {_last_emitted_segment_index}")
    
    # Third transcription: 3 segments total (simulating more buffer growth)
    print("\n3. Third transcription (3 segments, 1 new):")
    segments_3 = [
        {'text': 'Hello world', 'confidence': 90, 'start': 0, 'end': 2},
        {'text': 'how are you', 'confidence': 85, 'start': 2, 'end': 4},
        {'text': 'today', 'confidence': 80, 'start': 4, 'end': 5}
    ]
    new_segments = []
    for i, seg in enumerate(segments_3):
        condition = i > _last_emitted_segment_index
        print(f"   Segment {i}: '{seg['text']}' - Check: {i} > {_last_emitted_segment_index} = {condition}")
        if condition:
            new_segments.append((i, seg))
    
    print(f"   Result: {len(new_segments)} new segments emitted")
    
    print("\n" + "=" * 60)
    print("ISSUE IDENTIFIED!")
    print("=" * 60)
    print("""
The problem is that whisper.cpp returns ALL segments from the beginning
of the buffer each time. The filtering logic works correctly (it only
emits new indices), BUT there's a subtle bug:

When _last_emitted_segment_index is -1 (after silence reset):
- Check: 0 > -1 = True (segment 0 should be emitted)
- This should work!

But wait... let me check if segments are actually being returned by the model.
""")

def test_actual_transcription():
    """Test actual transcription if model is available."""
    print("\n" + "=" * 60)
    print("Testing Actual Transcription")
    print("=" * 60)
    
    try:
        processor = AccumulatingTranscriptionProcessor(
            model_size="tiny",
            window_size=60.0,
            update_frequency=2.0,
            silence_timeout=3.0
        )
        
        print("Loading model...")
        processor.load_model()
        print("Model loaded successfully!")
        
        # Generate test audio (1 second of 440Hz sine wave)
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5
        
        print(f"\nFeeding {duration}s of test audio...")
        processor.start()
        processor.feed_audio(audio)
        
        # Wait a bit
        import time
        time.sleep(0.5)
        
        # Check results
        results = processor.get_results()
        print(f"\nResults received: {len(results)}")
        for r in results:
            print(f"  - '{r.text}' (conf: {r.confidence}%)")
        
        processor.stop()
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_segment_filtering()
    test_actual_transcription()
