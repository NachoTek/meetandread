#!/usr/bin/env python3
"""
Test real microphone recording to check for speed/choppiness issues.

This script records a short clip from your microphone and analyzes it for:
- Sample rate consistency
- Duration accuracy
- Frame continuity (no gaps)
"""

import tempfile
import time
import wave
import numpy as np
from pathlib import Path
from meetandread.audio.session import AudioSession, SessionConfig, SourceConfig
from meetandread.audio.capture.devices import list_mic_inputs

def analyze_recording(wav_path, expected_duration):
    """Analyze a WAV file for recording issues."""
    print(f"\nAnalyzing: {wav_path.name}")
    print("-" * 70)

    with wave.open(str(wav_path), 'rb') as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        duration = n_frames / sample_rate

    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Channels: {n_channels}")
    print(f"  Sample width: {sample_width} bytes")
    print(f"  Frames: {n_frames}")
    print(f"  Duration: {duration:.4f} seconds")
    print(f"  Expected duration: {expected_duration:.4f} seconds")

    # Check duration accuracy
    duration_error = abs(duration - expected_duration)
    speedup = duration / expected_duration if expected_duration > 0 else 1.0

    print(f"\n  Duration error: {duration_error:.4f} seconds")
    print(f"  Speedup factor: {speedup:.3f}x")

    if duration_error > 0.1:
        print(f"  WARNING: Duration mismatch > 100ms!")
        print(f"    This could cause audio to sound sped up or slowed down.")

    # Check for gaps by analyzing audio continuity
    with wave.open(str(wav_path), 'rb') as wf:
        audio_data = wf.readframes(n_frames)
        int16_samples = np.frombuffer(audio_data, dtype=np.int16)

    # Convert to float32 for analysis
    float_samples = int16_samples.astype(np.float32) / 32767.0

    # Check for silent gaps (potential dropouts)
    if len(float_samples) > 0:
        # Calculate RMS in chunks
        chunk_size = int(sample_rate * 0.01)  # 10ms chunks
        chunks = [float_samples[i:i+chunk_size] for i in range(0, len(float_samples), chunk_size)]
        rms_values = [np.sqrt(np.mean(chunk**2)) for chunk in chunks if len(chunk) > 0]

        # Count silent chunks
        silent_threshold = 0.01
        silent_count = sum(1 for rms in rms_values if rms < silent_threshold)
        silent_percent = (silent_count / len(rms_values)) * 100 if rms_values else 0

        print(f"\n  Audio analysis:")
        print(f"    Total chunks (10ms each): {len(rms_values)}")
        print(f"    Silent chunks (<1% amplitude): {silent_count} ({silent_percent:.1f}%)")

        if silent_percent > 50:
            print(f"    WARNING: High percentage of silence!")
            print(f"      This could indicate dropped frames or microphone issues.")
        elif silent_percent > 20:
            print(f"    INFO: Moderate silence - may be normal if recording quiet room.")

    return {
        'sample_rate': sample_rate,
        'duration': duration,
        'expected_duration': expected_duration,
        'speedup': speedup,
        'duration_error': duration_error,
    }

def test_microphone_recording(duration_seconds=5.0):
    """Record from microphone and analyze the result."""
    print("=" * 70)
    print("REAL MICROPHONE RECORDING TEST")
    print("=" * 70)
    print(f"\nRecording duration: {duration_seconds} seconds")
    print(f"Target sample rate: 16000 Hz")

    # Get available microphones
    mics = list_mic_inputs()
    if not mics:
        print("\nERROR: No microphone devices found!")
        return False

    print(f"\nAvailable microphones:")
    for i, mic in enumerate(mics):
        print(f"  {i}: {mic['name']} ({mic['default_samplerate']} Hz, {mic['max_input_channels']} ch)")

    # Use first microphone
    selected_mic = mics[0]
    print(f"\nUsing microphone: {selected_mic['name']}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        print(f"\nRecording for {duration_seconds} seconds...")
        print("Please speak into the microphone now...")

        # Configure recording
        config = SessionConfig(
            sources=[SourceConfig(type='mic', device_id=selected_mic['index'])],
            output_dir=tmp_path,
            sample_rate=16000,
            channels=1,
        )

        # Record
        session = AudioSession()
        start_time = time.time()
        session.start(config)

        # Wait for recording to complete
        time.sleep(duration_seconds)

        wav_path = session.stop()
        actual_duration = time.time() - start_time

        print(f"\nRecording stopped.")
        print(f"Actual recording time: {actual_duration:.2f} seconds")

        # Analyze
        result = analyze_recording(wav_path, duration_seconds)

        print(f"\n" + "=" * 70)
        print("TEST RESULTS")
        print("=" * 70)

        if abs(result['speedup'] - 1.0) > 0.05:
            print(f"\nFAIL: Audio speed issue detected!")
            print(f"  Speedup factor: {result['speedup']:.3f}x")
            print(f"  (1.0 = correct, >1.0 = too fast, <1.0 = too slow)")
            print(f"\n  This causes the 'sped up' audio symptom.")
            return False
        else:
            print(f"\nPASS: Audio speed is correct (speedup = {result['speedup']:.3f}x)")

        print(f"\nPlayback the recorded file to check for choppiness:")
        print(f"  {wav_path}")
        print(f"\nIf the audio sounds:")
        print(f"  - Normal: The issue may be intermittent or device-specific")
        print(f"  - Sped up: There's a sample rate mismatch")
        print(f"  - Choppy: There may be frame drops or denoising issues")

        return True

if __name__ == "__main__":
    import sys

    duration = 5.0
    if len(sys.argv) > 1:
        try:
            duration = float(sys.argv[1])
        except ValueError:
            print(f"Invalid duration: {sys.argv[1]}")
            print(f"Usage: python test_mic_recording.py [duration_seconds]")
            sys.exit(1)

    success = test_microphone_recording(duration)
    sys.exit(0 if success else 1)
