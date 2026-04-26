#!/usr/bin/env python3
"""Diagnose audio input levels for speech detection."""

import sys
sys.path.insert(0, 'src')

import numpy as np
from meetandread.audio import AudioSession, SessionConfig, SourceConfig

def check_audio_levels():
    """Check audio input levels from microphone."""
    print("=" * 60)
    print("Audio Input Level Diagnostic")
    print("=" * 60)
    print("\nSpeak into your microphone for 5 seconds...")
    print("This will measure audio energy levels to adjust speech threshold.\n")

    # Create audio config for mic only
    config = SessionConfig(
        sources=[SourceConfig(type='mic', gain=1.0)]
    )

    # Collect audio samples
    energies = []
    samples_count = 0
    target_samples = 16000 * 5  # 5 seconds at 16kHz

    def collect_audio(audio_chunk):
        nonlocal energies, samples_count

        # Calculate energy (RMS) - same as processor
        if len(audio_chunk) > 0:
            energy = np.sqrt(np.mean(audio_chunk ** 2))
            energies.append(energy)
            samples_count += len(audio_chunk)

            # Show live level
            if len(energies) % 100 == 0:
                print(f"  Sample {len(energies)}: Energy = {energy:.6f} (threshold=0.01, speech={'YES' if energy > 0.01 else 'NO'})")

    # Create session with callback
    config.on_audio_frame = collect_audio

    print("Starting audio capture...")
    print("Current speech threshold: 0.01\n")

    import time
    session = AudioSession()

    try:
        session.start(config)

        # Record for 5 seconds
        start_time = time.time()
        while time.time() - start_time < 5:
            time.sleep(0.1)

        session.stop()

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    # Analyze results
    print("\n" + "=" * 60)
    print("Analysis Results")
    print("=" * 60)

    if not energies:
        print("\nERROR: No audio captured!")
        print("Possible issues:")
        print("  1. Microphone not selected or muted")
        print("  2. No permission to access microphone")
        print("  3. Audio device not working")
        return

    energies = np.array(energies)

    print(f"\nTotal samples analyzed: {len(energies)}")
    print(f"Min energy: {energies.min():.6f}")
    print(f"Max energy: {energies.max():.6f}")
    print(f"Mean energy: {energies.mean():.6f}")
    print(f"Median energy: {np.median(energies):.6f}")

    # Check speech detection
    speech_samples = np.sum(energies > 0.01)
    speech_percent = (speech_samples / len(energies)) * 100

    print(f"\nCurrent threshold: 0.01")
    print(f"Speech detected: {speech_samples}/{len(energies)} samples ({speech_percent:.1f}%)")

    if speech_percent < 10:
        print("\n⚠ WARNING: Speech detection threshold is TOO HIGH!")
        print("   Your microphone is quieter than the threshold.")
        print(f"   Suggested threshold: {energies.mean() * 0.5:.6f}")
        print("\n   Fix: Lower SPEECH_THRESHOLD in accumulating_processor.py")
        print("        Current location: Line 204")
        print("        Change: SPEECH_THRESHOLD = 0.01 → ~0.005 or lower")
    elif speech_percent > 90:
        print("\n⚠ WARNING: Speech detection threshold is TOO LOW!")
        print("   Background noise is being detected as speech.")
        print(f"   Suggested threshold: {energies.mean() * 1.5:.6f}")
    else:
        print("\n✓ Speech threshold looks reasonable")

if __name__ == "__main__":
    check_audio_levels()
