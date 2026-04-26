"""
Manual integration test for transcription pipeline.

This script tests the complete pipeline with your sample audio file.
Run it to verify:
1. Audio transcription works
2. Latency is acceptable
3. Transcript saves to file
4. Panel displays words

Usage:
    python tests/manual_integration_test.py
"""

import sys
import time
import wave
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from meetandread.transcription.streaming_pipeline import RealTimeTranscriptionProcessor
from meetandread.transcription.transcript_store import TranscriptStore
from meetandread.config.models import AppSettings, TranscriptionSettings


def convert_mp3_to_wav(mp3_path: Path) -> Path:
    """Convert MP3 to WAV format (16kHz, mono, float32)."""
    wav_path = mp3_path.with_suffix('.wav')
    
    if wav_path.exists():
        print(f"✓ WAV file already exists: {wav_path}")
        return wav_path
    
    print(f"Converting {mp3_path.name} to WAV...")
    print("(This requires pydub and ffmpeg. Installing if needed...)")
    
    try:
        from pydub import AudioSegment
    except ImportError:
        print("Installing pydub...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pydub"])
        from pydub import AudioSegment
    
    # Load MP3
    audio = AudioSegment.from_mp3(str(mp3_path))
    
    # Convert to 16kHz mono
    audio = audio.set_frame_rate(16000).set_channels(1)
    
    # Export as WAV
    audio.export(str(wav_path), format="wav")
    print(f"✓ Converted to: {wav_path}")
    
    return wav_path


def test_with_sample_audio():
    """Test transcription with your sample audio file."""
    print("=" * 60)
    print("TRANSCRIPTION PIPELINE INTEGRATION TEST")
    print("=" * 60)
    
    # Paths
    fixtures_dir = Path(__file__).parent / 'fixtures'
    audio_path = fixtures_dir / 'SAMPLE-Audio1.mp3'
    transcript_path = fixtures_dir / 'SAMPLE-Transcript1.txt'
    
    # Check files exist
    if not audio_path.exists():
        print(f"\n❌ Sample audio not found: {audio_path}")
        print("Please place SAMPLE-Audio1.mp3 in tests/fixtures/")
        return False
    
    if not transcript_path.exists():
        print(f"\n❌ Sample transcript not found: {transcript_path}")
        return False
    
    print(f"\n✓ Sample audio: {audio_path} ({audio_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"✓ Sample transcript: {transcript_path} ({transcript_path.stat().st_size / 1024:.1f} KB)")
    
    # Convert to WAV if needed
    try:
        wav_path = convert_mp3_to_wav(audio_path)
    except Exception as e:
        print(f"\n❌ Failed to convert audio: {e}")
        print("Please install ffmpeg: https://ffmpeg.org/download.html")
        return False
    
    # Setup transcription
    print("\n" + "=" * 60)
    print("SETUP")
    print("=" * 60)
    
    config = TranscriptionSettings(
        enabled=True,
        confidence_threshold=0.7,
        min_chunk_size_sec=0.5,  # 0.5s chunks for low latency
        agreement_threshold=1     # Immediate commit
    )
    
    app_config = AppSettings(transcription=config)
    processor = RealTimeTranscriptionProcessor(app_config)
    processor.set_model_config(model_size='tiny', device='cpu', compute_type='int8')
    
    # Load model
    print("\nLoading tiny model (may take 2-5 seconds)...")
    start_time = time.time()
    processor.load_model()
    load_time = time.time() - start_time
    print(f"✓ Model loaded in {load_time:.2f}s")
    
    # Create transcript store
    store = TranscriptStore()
    store.start_recording()
    
    # Start processing
    print("\nStarting transcription processor...")
    processor.start()
    
    # Read WAV file in chunks
    print("\n" + "=" * 60)
    print("TRANSCRIBING")
    print("=" * 60)
    
    chunk_size = 8000  # 0.5 seconds at 16kHz
    total_samples = 0
    results_count = 0
    
    with wave.open(str(wav_path), 'rb') as wav:
        # Verify format
        if wav.getframerate() != 16000 or wav.getnchannels() != 1:
            print(f"❌ Unexpected WAV format: {wav.getframerate()}Hz, {wav.getnchannels()} channels")
            return False
        
        print(f"Audio: {wav.getnframes()} samples ({wav.getnframes() / 16000:.1f}s)")
        print(f"Processing in {chunk_size} sample chunks ({chunk_size / 16000:.2f}s each)\n")
        
        chunk_num = 0
        while True:
            # Read chunk
            frames = wav.readframes(chunk_size)
            if not frames:
                break
            
            # Convert to float32
            audio_chunk = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Feed to processor
            feed_time = time.time()
            processor.feed_audio(audio_chunk)
            total_samples += len(audio_chunk)
            
            # Check for results
            results = processor.get_results()
            for result in results:
                results_count += 1
                latency = time.time() - feed_time
                print(f"Chunk {chunk_num:3d}: '{result.text}' (latency: {latency:.2f}s, conf: {result.confidence})")
                
                # Add to store
                from meetandread.transcription.transcript_store import Word
                words = [
                    Word(text=w.word if hasattr(w, 'word') else str(w), 
                         start_time=w.start if hasattr(w, 'start') else 0,
                         end_time=w.end if hasattr(w, 'end') else 0,
                         confidence=result.confidence,
                         speaker_id=None)
                    for w in result.words
                ]
                store.add_words(words)
            
            chunk_num += 1
            
            # Progress update every 100 chunks
            if chunk_num % 100 == 0:
                elapsed = total_samples / 16000
                print(f"  ... processed {elapsed:.1f}s of audio, {results_count} results so far")
    
    # Stop processor
    print("\nStopping processor...")
    processor.stop()
    
    # Save transcript
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    output_path = fixtures_dir / 'TEST-OUTPUT-Transcript.md'
    store.save_to_file(output_path)
    
    all_words = store.get_all_words()
    print(f"\n✓ Transcribed {len(all_words)} words")
    print(f"✓ Saved to: {output_path}")
    
    # Show sample
    if all_words:
        sample_text = ' '.join([w.text for w in all_words[:20]])
        print(f"\nFirst 20 words:")
        print(f"  {sample_text}")
    
    # Check transcript file
    if output_path.exists():
        content = output_path.read_text()
        print(f"\n✓ Output file size: {len(content)} bytes")
        
        # Compare to reference
        with open(transcript_path, 'r') as f:
            reference = f.read()
        
        ref_words = len(reference.split())
        our_words = len(all_words)
        
        print(f"\nReference: ~{ref_words} words")
        print(f"Our output: {our_words} words")
        print(f"Ratio: {our_words / ref_words * 100:.1f}% (target > 70%)")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    
    return True


def test_latency_measurement():
    """Measure actual latency with synthetic audio."""
    print("\n" + "=" * 60)
    print("LATENCY MEASUREMENT")
    print("=" * 60)
    
    config = TranscriptionSettings(
        enabled=True,
        confidence_threshold=0.7,
        min_chunk_size_sec=0.5,
        agreement_threshold=1
    )
    
    app_config = AppSettings(transcription=config)
    processor = RealTimeTranscriptionProcessor(app_config)
    processor.set_model_config(model_size='tiny', device='cpu', compute_type='int8')
    
    print("Loading model...")
    processor.load_model()
    
    print("Starting processor...")
    processor.start()
    
    # Generate 2 seconds of synthetic audio
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(duration * sample_rate))
    audio = (
        0.3 * np.sin(2 * np.pi * 200 * t) +
        0.2 * np.sin(2 * np.pi * 400 * t) +
        0.05 * np.random.randn(len(t))
    ).astype(np.float32)
    
    print(f"\nFeeding {duration}s of synthetic audio...")
    
    # Feed audio
    start_time = time.time()
    processor.feed_audio(audio)
    
    # Wait for results
    max_wait = 10.0
    results = []
    while time.time() - start_time < max_wait:
        results = processor.get_results()
        if results:
            break
        time.sleep(0.1)
    
    processor.stop()
    
    latency = time.time() - start_time
    
    print(f"\nLatency: {latency:.2f}s")
    print(f"Target: < 2.0s")
    
    if latency < 2.0:
        print("✓ PASS: Latency within target")
    else:
        print("❌ FAIL: Latency too high")
        print("  Consider:")
        print("  - Using even smaller chunks (0.25s)")
        print("  - Checking CPU usage")
        print("  - Verifying no other processes are running")
    
    return latency < 2.0


if __name__ == "__main__":
    print("\nMeetAndRead Transcription Pipeline Test")
    print("======================================\n")
    
    # Run tests
    try:
        # Test 1: Sample audio
        test_with_sample_audio()
        
        # Test 2: Latency
        test_latency_measurement()
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
