"""
Manual integration test for transcription pipeline.

This script tests the complete pipeline with your sample audio file.
Run it to verify:
1. Audio transcription works
2. Transcript saves to file
3. Panel displays words

Usage:
    python tests/manual_integration_test.py

Note: The legacy RealTimeTranscriptionProcessor was removed in the
streaming_pipeline.py cleanup (T03).  This script now tests only the
TranscriptStore file-saving path.  Full end-to-end latency testing
should use the active recording controller.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from meetandread.transcription.transcript_store import TranscriptStore, Word
from meetandread.config.models import AppSettings, TranscriptionSettings


def test_transcript_store_with_sample_words():
    """Test transcript store saves correctly."""
    print("=" * 60)
    print("TRANSCRIPT STORE INTEGRATION TEST")
    print("=" * 60)

    # Create store and add words
    store = TranscriptStore()
    store.start_recording()

    words = [
        Word(text="Hello", start_time=0.0, end_time=0.5, confidence=85),
        Word(text="world", start_time=0.6, end_time=1.0, confidence=92),
    ]
    store.add_words(words)

    # Save to file
    output_path = Path(__file__).parent / 'fixtures' / 'TEST-OUTPUT-Transcript.md'
    store.save_to_file(output_path)

    if output_path.exists():
        content = output_path.read_text()
        print(f"\n✓ Transcript saved to: {output_path}")
        print(f"✓ File size: {len(content)} bytes")
        print(f"✓ Content preview: {content[:100]}...")
        return True
    else:
        print(f"\n❌ Failed to save transcript")
        return False


if __name__ == "__main__":
    print("\nMeetAndRead Transcript Store Integration Test")
    print("=============================================\n")

    try:
        result = test_transcript_store_with_sample_words()
        if result:
            print("\n✓ All tests passed")
        else:
            print("\n❌ Test failed")
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
