"""Regression test for CLI fake recording duration.

Verifies that `python -m meetandread.audio.cli record --fake <wav> --seconds N`
produces an output WAV of approximately N seconds, even when the fake source
can emit audio faster than real-time.
"""

import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

# These tests spawn the CLI as a subprocess, which imports ``sounddevice`` and
# needs the real PortAudio library. Marked `windows` so the authoritative pass
# runs them under the Windows venv (ADR 0001) and they skip off-Windows.
pytestmark = pytest.mark.windows


def generate_test_wav(path: Path, duration_seconds: float, sample_rate: int = 16000) -> None:
    """Generate a sine wave WAV file of specified duration.
    
    Args:
        path: Output path for the WAV file
        duration_seconds: Duration of the audio in seconds
        sample_rate: Sample rate in Hz
    """
    # Generate sine wave
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    frequency = 440  # A4 note
    amplitude = 0.5
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Convert to int16
    int16_data = (sine_wave * 32767).astype(np.int16)
    
    # Write WAV file
    with wave.open(str(path), 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(int16_data.tobytes())


def get_wav_duration(path: Path) -> float:
    """Get the duration of a WAV file in seconds."""
    with wave.open(str(path), 'rb') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        return frames / rate


def test_cli_fake_seconds_truncates_output(tmp_path: Path) -> None:
    """Test that --seconds N produces output of ~N seconds, not full input duration.
    
    This is a regression test for the issue where fake audio sources emitting
    faster than real-time would result in the full input file being copied,
    ignoring the --seconds parameter.
    
    Test scenario:
    - Generate a 9-second input WAV
    - Run CLI with --seconds 5
    - Assert output is ~5 seconds (not 9 seconds)
    """
    # Generate 9-second test input WAV
    input_wav = tmp_path / "input_9s.wav"
    generate_test_wav(input_wav, duration_seconds=9.0, sample_rate=16000)
    
    # Verify input file properties
    with wave.open(str(input_wav), 'rb') as wav_in:
        input_frames = wav_in.getnframes()
        input_rate = wav_in.getframerate()
        input_channels = wav_in.getnchannels()
    
    assert input_rate == 16000, f"Expected 16000 Hz, got {input_rate}"
    assert input_channels == 1, f"Expected mono, got {input_channels} channels"
    assert input_frames == int(9.0 * 16000), f"Expected {int(9.0 * 16000)} frames, got {input_frames}"
    
    # Create output directory
    output_dir = tmp_path / "recordings"
    output_dir.mkdir()
    
    # Run CLI: 9s input + --seconds 5 should output ~5s
    repo_root = Path(__file__).parent.parent
    cmd = [
        sys.executable, '-m', 'meetandread.audio.cli',
        'record',
        '--fake', str(input_wav),
        '--seconds', '5',
        '--output-dir', str(output_dir),
    ]
    
    env = {
        'PYTHONPATH': str(repo_root / 'src'),
    }
    
    result = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=30,
        env={**dict(subprocess.os.environ), **env},
    )
    
    # Assert CLI succeeded
    assert result.returncode == 0, (
        f"CLI failed with return code {result.returncode}\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    
    # Find the output WAV file (newest .wav in output_dir)
    wav_files = list(output_dir.glob('*.wav'))
    assert len(wav_files) > 0, f"No WAV files found in {output_dir}"
    
    # Get the most recently created WAV file
    output_wav = max(wav_files, key=lambda p: p.stat().st_mtime)
    
    # Verify output WAV properties
    with wave.open(str(output_wav), 'rb') as wav_out:
        output_frames = wav_out.getnframes()
        output_rate = wav_out.getframerate()
        output_channels = wav_out.getnchannels()
    
    # Expected frames for 5 seconds at 16000 Hz
    expected_frames = int(round(5 * 16000))
    
    # Assertions
    assert output_rate == 16000, f"Expected output rate 16000 Hz, got {output_rate}"
    assert output_channels == 1, f"Expected mono output, got {output_channels} channels"
    
    # Key regression assertion: output should be capped to requested duration
    assert output_frames == expected_frames, (
        f"Output duration mismatch: expected {expected_frames} frames (~5s), "
        f"got {output_frames} frames ({output_frames / 16000:.2f}s). "
        f"This indicates the max_frames cap is not working - full file was copied."
    )
    
    # Additional safety: output should be strictly less than input
    assert output_frames < input_frames, (
        f"Output ({output_frames} frames) should be less than input ({input_frames} frames). "
        f"Full copy regression detected."
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
