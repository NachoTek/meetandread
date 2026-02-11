"""
Test script for inducing low confidence segments.

This script uses FakeAudioModule with noise to generate audio that
will trigger lower confidence scores from the Whisper model.

Usage:
    python test_low_confidence.py [--wav PATH] [--noise LEVEL] [--duration SECONDS]

Examples:
    # Moderate noise (expect 50-70% confidence)
    python test_low_confidence.py --wav test.wav --noise 0.3

    # High noise (expect 30-50% confidence)
    python test_low_confidence.py --wav test.wav --noise 0.7

    # Extreme noise (expect <30% confidence)
    python test_low_confidence.py --wav test.wav --noise 1.0
"""

import argparse
import numpy as np
import wave
from pathlib import Path
import sys

# Only import FakeAudioModule when not just showing guide
FakeAudioModule = None
if not '--guide-only' in sys.argv:
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    from metamemory.audio.capture.fake_module import FakeAudioModule
    import queue
    import time


def generate_test_speech_wav(output_path: str, duration: int = 10.0) -> None:
    """
    Generate a simple test WAV file with spoken text.
    
    Creates a WAV file with synthesized speech-like audio for testing.
    """
    sample_rate = 16000
    channels = 1
    
    # Generate a simple tone sequence (simulates speech patterns)
    # This is basic - in real testing, use actual recorded speech
    duration_samples = int(duration * sample_rate)
    
    # Generate varying frequencies to simulate speech patterns
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Mix of frequencies to simulate speech-like audio
    audio_data = (
        0.5 * np.sin(2 * np.pi * 200 * t) +  # Base
        0.3 * np.sin(2 * np.pi * 400 * t) +  # Harmonics
        0.2 * np.sin(2 * np.pi * 600 * t) +
        0.1 * np.random.normal(0, 0.1, t.shape)  # Some randomness
    )
    
    # Add amplitude variation to simulate speech dynamics
    envelope = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t))
    audio_data = audio_data * envelope
    
    # Convert to int16
    audio_int16 = (audio_data * 32767).astype(np.int16)
    
    # Write WAV file
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    
    print(f"Generated test WAV: {output_path} ({duration}s)")


def test_noise_levels(wav_path: str) -> None:
    """
    Test different noise levels to see confidence scores.
    
    This requires metamemory to be running with FakeAudioModule.
    """
    print("\n" + "="*70)
    print("NOISE LEVEL GUIDE FOR LOW CONFIDENCE TESTING")
    print("="*70)
    print()
    print("Noise Level | Expected Confidence | Enhancement Behavior")
    print("-" * 70)
    print("0.0 (clean) | 70-100%          | No enhancement (all above 70% threshold)")
    print("0.3 (light)  | 50-70%           | Some enhancement (below 70% threshold)")
    print("0.5 (moderate) | 40-60%          | Moderate enhancement")
    print("0.7 (high)    | 20-40%          | Heavy enhancement")
    print("1.0 (extreme) | <30%              | Near-complete enhancement")
    print()
    print("USAGE IN METAMEMORY:")
    print("-" * 70)
    print("1. Start metamemory with FakeAudioModule enabled")
    print("2. Use settings panel to verify enhancement is enabled")
    print("3. Check enhancement threshold (default: 70%)")
    print("4. Speak naturally to generate baseline confidence")
    print("5. Observe: segments below 70% should appear in bold (enhanced)")
    print()
    print("TO INDUCE LOW CONFIDENCE:")
    print("-" * 70)
    print("Method 1: Use FakeAudioModule with noise_level")
    print("  Modify source code to pass noise_level to FakeAudioModule")
    print("  noise_level=0.5 for moderate, 0.7 for high noise")
    print()
    print("Method 2: Mumble or speak quietly")
    print("  Mumbling naturally lowers confidence scores")
    print("  Distance from microphone affects confidence")
    print()
    print("Method 3: Add background noise")
    print("  Play music/radio at low volume while speaking")
    print("  Whisper models are sensitive to background noise")
    print()
    print("Method 4: Speak quickly or unclearly")
    print("  Fast speech causes more transcription errors")
    print("  Unclear pronunciation triggers enhancement")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Test low confidence generation for enhancement testing"
    )
    parser.add_argument(
        '--wav',
        type=str,
        default='test_speech.wav',
        help='WAV file to generate or test'
    )
    parser.add_argument(
        '--noise',
        type=float,
        choices=[0.0, 0.3, 0.5, 0.7, 1.0],
        default=0.5,
        help='Noise level to add (0.0-1.0)'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=10.0,
        help='Duration in seconds for generated test audio'
    )
    parser.add_argument(
        '--guide-only',
        action='store_true',
        help='Only show the noise level guide'
    )
    parser.add_argument(
        '--generate-only',
        action='store_true',
        help='Only generate test WAV, do not test'
    )
    
    args = parser.parse_args()
    
    if args.guide_only:
        test_noise_levels(args.wav)
        return
    
    if args.generate_only:
        # Generate test WAV
        wav_path = Path(args.wav)
        generate_test_speech_wav(str(wav_path), args.duration)
        print(f"\nGenerated: {wav_path}")
        print(f"Use with: python test_low_confidence.py --wav {args.wav} --noise {args.noise}")
        return
    
    # Test with noise
    print(f"\nTesting FakeAudioModule with noise_level={args.noise}")
    print(f"WAV file: {args.wav}")
    print(f"Duration: {Path(args.wav).stat().st_size / (16000 * 2):.1f} seconds" if Path(args.wav).exists() else "N/A")
    
    # Initialize FakeAudioModule with noise
    fake_module = FakeAudioModule(
        wav_path=args.wav,
        noise_level=args.noise,
        loop=True
    )
    
    print(f"\nFakeAudioModule metadata:")
    metadata = fake_module.get_metadata()
    for key, value in metadata.items():
        print(f"  {key}: {value}")
    
    print(f"\nExpected confidence range: {int(70 - args.noise * 70)}% - {int(100 - args.noise * 30)}%")
    print(f"\nNote: To use with metamemory, modify the code to instantiate")
    print(f"      FakeAudioModule with noise_level parameter.")


if __name__ == "__main__":
    main()
