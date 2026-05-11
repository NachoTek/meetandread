#!/usr/bin/env python3
"""
Test to verify that audio resampling preserves correct duration.

This test creates a 48kHz WAV file, records it at 16kHz, and verifies
that the output has the correct duration (not sped up).
"""

import tempfile
from pathlib import Path
import numpy as np
import wave
import time
from meetandread.audio.session import AudioSession, SessionConfig, SourceConfig

def create_sine_wave_wav(path, frequency=440.0, duration=0.5, sample_rate=16000, channels=1, amplitude=0.5):
    """Create a sine wave WAV file for testing."""
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)
    int16_data = (sine_wave * 32767).astype(np.int16)

    if channels == 2:
        int16_data = np.column_stack([int16_data, int16_data])

    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_data.tobytes())

    return num_samples

def read_wav_file(path):
    """Read a WAV file and return its properties."""
    with wave.open(str(path), 'rb') as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        duration = n_frames / sample_rate
        return n_channels, sample_rate, sample_width, n_frames, duration

def test_resampling_preserves_duration():
    """Test that 48kHz -> 16kHz resampling preserves duration."""
    print("Testing: 48kHz -> 16kHz resampling preserves duration")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create a 0.5 second WAV file at 48kHz
        original_duration = 0.5
        original_sample_rate = 48000
        target_sample_rate = 16000
        test_wav = tmp_path / "test_48k.wav"

        print(f"\n1. Creating test WAV file:")
        print(f"   Duration: {original_duration} seconds")
        print(f"   Sample rate: {original_sample_rate} Hz")
        print(f"   Expected samples: {int(original_sample_rate * original_duration)}")

        create_sine_wave_wav(
            test_wav,
            frequency=880.0,
            duration=original_duration,
            sample_rate=original_sample_rate,
            channels=2,  # Stereo
        )

        # Verify the test WAV
        n_ch, sr, sw, n_fr, dur = read_wav_file(test_wav)
        print(f"\n2. Test WAV file created:")
        print(f"   Channels: {n_ch}")
        print(f"   Sample rate: {sr} Hz")
        print(f"   Frames: {n_fr}")
        print(f"   Duration: {dur:.4f} seconds")

        assert sr == original_sample_rate, f"Test WAV has wrong sample rate: {sr}"
        assert abs(dur - original_duration) < 0.01, f"Test WAV has wrong duration: {dur}"

        # Record the audio at 16kHz
        print(f"\n3. Recording to {target_sample_rate} Hz target rate...")

        config = SessionConfig(
            sources=[SourceConfig(type='fake', fake_path=str(test_wav))],
            output_dir=tmp_path,
            sample_rate=target_sample_rate,
            channels=1,
        )

        session = AudioSession()
        session.start(config)

        # Wait for recording to complete
        time.sleep(0.6)  # Slightly longer than audio duration

        wav_path = session.stop()

        # Verify the output
        print(f"\n4. Output WAV file:")
        out_n_ch, out_sr, out_sw, out_n_fr, out_dur = read_wav_file(wav_path)
        print(f"   Channels: {out_n_ch}")
        print(f"   Sample rate: {out_sr} Hz")
        print(f"   Frames: {out_n_fr}")
        print(f"   Duration: {out_dur:.4f} seconds")

        print(f"\n5. Verification:")
        print(f"   Target sample rate: {target_sample_rate} Hz")
        print(f"   Actual sample rate: {out_sr} Hz")
        print(f"   Sample rate correct: {'PASS' if out_sr == target_sample_rate else 'FAIL'}")

        print(f"\n   Original duration: {original_duration:.4f} seconds")
        print(f"   Output duration: {out_dur:.4f} seconds")
        print(f"   Duration correct: {'PASS' if abs(out_dur - original_duration) < 0.01 else 'FAIL'}")

        # Calculate speedup factor
        speedup = out_dur / original_duration if original_duration > 0 else 1.0
        print(f"\n   Speedup factor: {speedup:.2f}x")
        print(f"   (1.0x = correct, 2.0x = plays twice as fast, 0.5x = plays half speed)")

        if abs(speedup - 1.0) > 0.05:
            print(f"\n   FAIL: Audio is sped up or slowed down!")
            print(f"   This causes the user's reported symptoms.")
            return False
        else:
            print(f"\n   PASS: Resampling preserves duration correctly")
            return True

if __name__ == "__main__":
    success = test_resampling_preserves_duration()
    sys.exit(0 if success else 1)
