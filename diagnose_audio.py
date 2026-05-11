#!/usr/bin/env python3
"""
Comprehensive audio pipeline diagnostic tool.

This script helps diagnose issues with sped-up or choppy audio recordings.
It checks:
1. Source device sample rates
2. Recording metadata
3. WAV file headers
4. Frame counts and durations
5. Potential sample rate mismatches
"""

import sys
import json
import argparse
from pathlib import Path

def check_audio_devices():
    """Check available audio devices and their sample rates."""
    print("=" * 70)
    print("AUDIO DEVICES")
    print("=" * 70)

    try:
        from meetandread.audio.capture.devices import list_mic_inputs
        devices = list_mic_inputs()

        if not devices:
            print("No microphone devices found")
            return

        for d in devices:
            print(f"\nDevice {d['index']}: {d['name']}")
            print(f"  Sample rate: {d['default_samplerate']} Hz")
            print(f"  Channels: {d['max_input_channels']}")
            print(f"  WASAPI: {d.get('hostapi') == 'Windows WASAPI'}")
    except Exception as e:
        print(f"Error checking devices: {e}")


def check_most_recent_recording(recording_dir=None):
    """Check the most recent recording for issues."""
    print("\n" + "=" * 70)
    print("MOST RECENT RECORDING")
    print("=" * 70)

    if recording_dir is None:
        from meetandread.audio.storage.paths import get_recordings_dir
        recording_dir = get_recordings_dir()

    recording_dir = Path(recording_dir)
    if not recording_dir.exists():
        print(f"Recording directory does not exist: {recording_dir}")
        return

    # Find most recent metadata file
    metadata_files = list(recording_dir.glob("*.pcm.part.json"))
    if not metadata_files:
        print("No recording metadata found")
        print(f"  Looking in: {recording_dir}")
        return

    latest = max(metadata_files, key=lambda p: p.stat().st_mtime)
    print(f"\nChecking: {latest.name}")
    print(f"  Created: {latest.stat().st_mtime}")
    print("-" * 70)

    # Load metadata
    with open(latest) as f:
        metadata = json.load(f)

    print(f"\nMETADATA (from JSON sidecar):")
    print(f"  Sample rate: {metadata['sample_rate']} Hz")
    print(f"  Channels: {metadata['channels']}")
    print(f"  Sample width: {metadata['sample_width_bytes']} bytes")

    # Check PCM part file
    part_file = latest.parent / latest.stem.replace('.json', '')
    if part_file.exists():
        part_size = part_file.stat().st_size
        bytes_per_frame = metadata['channels'] * metadata['sample_width_bytes']
        expected_frames = part_size // bytes_per_frame
        duration_at_meta_rate = expected_frames / metadata['sample_rate']
        print(f"\nPCM PART FILE:")
        print(f"  Size: {part_size} bytes")
        print(f"  Expected frames: {expected_frames}")
        print(f"  Duration (at metadata rate): {duration_at_meta_rate:.2f} seconds")
    else:
        print(f"\nPCM part file not found: {part_file.name}")

    # Check WAV file
    wav_file = latest.parent / latest.stem.replace('.pcm.json', '.wav')
    if not wav_file.exists():
        # Try other naming pattern
        wav_file = latest.parent / latest.stem.replace('.part.json', '.wav')

    if wav_file.exists():
        print(f"\nWAV FILE:")
        print(f"  Path: {wav_file.name}")

        # Check WAV header
        try:
            import wave
            with wave.open(str(wav_file), 'rb') as wav:
                wav_rate = wav.getframerate()
                wav_channels = wav.getnchannels()
                wav_width = wav.getsampwidth()
                wav_frames = wav.getnframes()
                wav_duration = wav_frames / wav_rate

                print(f"  Sample rate: {wav_rate} Hz")
                print(f"  Channels: {wav_channels}")
                print(f"  Sample width: {wav_width} bytes")
                print(f"  Frames: {wav_frames}")
                print(f"  Duration: {wav_duration:.2f} seconds")

            # Check for mismatches
            print(f"\nDIAGNOSTICS:")
            issues = []

            if metadata['sample_rate'] != wav_rate:
                issue = f"❌ SAMPLE RATE MISMATCH!"
                print(f"  {issue}")
                print(f"     Metadata: {metadata['sample_rate']} Hz")
                print(f"     WAV header: {wav_rate} Hz")
                print(f"     Impact: Audio will play at {wav_rate / metadata['sample_rate']:.2f}x speed")
                issues.append("sample_rate_mismatch")
            else:
                print(f"  ✅ Sample rate matches")

            if metadata['channels'] != wav_channels:
                issue = f"⚠️  CHANNEL MISMATCH!"
                print(f"  {issue}")
                print(f"     Metadata: {metadata['channels']}")
                print(f"     WAV header: {wav_channels}")
                issues.append("channel_mismatch")
            else:
                print(f"  ✅ Channels match")

            if metadata['sample_width_bytes'] != wav_width:
                issue = f"⚠️  SAMPLE WIDTH MISMATCH!"
                print(f"  {issue}")
                print(f"     Metadata: {metadata['sample_width_bytes']} bytes")
                print(f"     WAV header: {wav_width} bytes")
                issues.append("sample_width_mismatch")
            else:
                print(f"  ✅ Sample width matches")

            # Check PCM vs WAV frame count
            if part_file.exists():
                pcm_frames = expected_frames
                if abs(pcm_frames - wav_frames) > 100:  # Allow small rounding errors
                    issue = f"⚠️  FRAME COUNT MISMATCH!"
                    print(f"  {issue}")
                    print(f"     PCM file: {pcm_frames} frames")
                    print(f"     WAV file: {wav_frames} frames")
                    print(f"     Difference: {pcm_frames - wav_frames} frames")
                    issues.append("frame_count_mismatch")
                else:
                    print(f"  ✅ Frame count matches")

            if issues:
                print(f"\n⚠️  ISSUES FOUND: {', '.join(issues)}")
                print(f"\nRECOMMENDATION:")
                if "sample_rate_mismatch" in issues:
                    print(f"  The sped-up audio is caused by the sample rate mismatch.")
                    print(f"  Check the AudioSession._writer creation in session.py")
                    print(f"  and ensure it uses config.sample_rate consistently.")
            else:
                print(f"\n✅ No structural issues found in recording")
                print(f"  If audio still sounds wrong, the issue may be in the")
                print(f"  audio capture or processing pipeline.")

        except Exception as e:
            print(f"  Error reading WAV file: {e}")
    else:
        print(f"\nWAV file not found: {wav_file.name}")
        print(f"  Recording may not be finalized yet")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose audio recording pipeline issues"
    )
    parser.add_argument(
        "--dir",
        help="Recording directory (default: Documents/meetandread/recordings)"
    )
    parser.add_argument(
        "--devices",
        action="store_true",
        help="Show audio device information"
    )
    parser.add_argument(
        "--recording",
        action="store_true",
        help="Show most recent recording information"
    )
    args = parser.parse_args()

    # Default to showing both if nothing specified
    if not args.devices and not args.recording:
        args.devices = True
        args.recording = True

    if args.devices:
        check_audio_devices()

    if args.recording:
        check_most_recent_recording(args.dir)


if __name__ == "__main__":
    main()
