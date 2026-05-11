#!/usr/bin/env python3
"""Debug script to check audio recording pipeline for speed/choppiness issues."""

import sys
import json
from pathlib import Path

def check_recording_metadata(recording_dir=None):
    """Check the most recent recording's metadata."""
    if recording_dir is None:
        from meetandread.audio.storage.paths import get_recordings_dir
        recording_dir = get_recordings_dir()

    # Find most recent metadata file
    metadata_files = list(Path(recording_dir).glob("*.pcm.part.json"))
    if not metadata_files:
        print("No recording metadata found")
        return

    latest = max(metadata_files, key=lambda p: p.stat().st_mtime)
    print(f"Checking: {latest.name}")
    print("=" * 60)

    with open(latest) as f:
        metadata = json.load(f)

    print(f"Sample rate: {metadata['sample_rate']} Hz")
    print(f"Channels: {metadata['channels']}")
    print(f"Sample width: {metadata['sample_width_bytes']} bytes")

    # Check corresponding WAV file
    wav_file = latest.parent / latest.stem.replace('.pcm', '.wav')
    if wav_file.exists():
        print(f"\nWAV file exists: {wav_file.name}")

        # Check WAV header
        import wave
        with wave.open(str(wav_file), 'rb') as wav:
            print(f"WAV sample rate: {wav.getframerate()} Hz")
            print(f"WAV channels: {wav.getnchannels()}")
            print(f"WAV sample width: {wav.getsampwidth()} bytes")
            print(f"WAV frames: {wav.getnframes()}")
            print(f"WAV duration: {wav.getnframes() / wav.getframerate():.2f} seconds")

        # Check for sample rate mismatch
        if metadata['sample_rate'] != wav.getframerate():
            print("\n⚠️  SAMPLE RATE MISMATCH DETECTED!")
            print(f"   Metadata says: {metadata['sample_rate']} Hz")
            print(f"   WAV header says: {wav.getframerate()} Hz")
            print("   This causes audio to play at wrong speed!")
    else:
        print(f"\nWAV file not found (not finalized yet): {wav_file.name}")

    # Check PCM part file size
    part_file = latest.parent / latest.stem.replace('.json', '')
    if part_file.exists():
        part_size = part_file.stat().st_size
        bytes_per_frame = metadata['channels'] * metadata['sample_width_bytes']
        expected_frames = part_size // bytes_per_frame
        duration_at_meta_rate = expected_frames / metadata['sample_rate']
        print(f"\nPCM part file: {part_size} bytes ({expected_frames} frames)")
        print(f"At metadata rate ({metadata['sample_rate']} Hz): {duration_at_meta_rate:.2f} seconds")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="Recording directory")
    args = parser.parse_args()
    check_recording_metadata(args.dir)
