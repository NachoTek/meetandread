"""CLI smoke harness for audio recording without UI.

Provides command-line interface for recording audio from various sources.
Useful for testing the recording pipeline without running the full GUI.

Examples:
    # Record from microphone for 5 seconds
    python -m metamemory.audio.cli record --mic

    # Record from system audio for 10 seconds  
    python -m metamemory.audio.cli record --system --seconds 10

    # Record both mic and system simultaneously
    python -m metamemory.audio.cli record --both

    # Record using a fake audio file (for testing)
    python -m metamemory.audio.cli record --fake /path/to/test.wav
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path for imports
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from metamemory.audio import (
    AudioSession,
    SessionConfig,
    SourceConfig,
    list_mic_inputs,
    list_loopback_outputs,
)
from metamemory.audio.capture import AudioSourceError


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog='metamemory.audio.cli',
        description='Audio recording CLI for metamemory',
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Record command
    record_parser = subparsers.add_parser(
        'record',
        help='Record audio from selected source(s)',
    )
    
    # Source selection (mutually exclusive)
    source_group = record_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        '--mic',
        action='store_true',
        help='Record from microphone',
    )
    source_group.add_argument(
        '--system',
        action='store_true',
        help='Record from system audio output (Windows only)',
    )
    source_group.add_argument(
        '--both',
        action='store_true',
        help='Record from both microphone and system audio',
    )
    source_group.add_argument(
        '--fake',
        metavar='PATH',
        type=str,
        help='Record using fake audio file (for testing)',
    )
    
    # Optional parameters
    record_parser.add_argument(
        '--seconds',
        type=float,
        default=5.0,
        help='Recording duration in seconds (default: 5)',
    )
    record_parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for recording (default: ~/Documents/metamemory)',
    )
    
    return parser


def check_mic_available() -> bool:
    """Check if any microphone devices are available."""
    try:
        devices = list_mic_inputs()
        return len(devices) > 0
    except Exception:
        return False


def check_system_available() -> bool:
    """Check if any system audio loopback devices are available."""
    try:
        devices = list_loopback_outputs()
        return len(devices) > 0
    except Exception:
        return False


def cmd_record(args) -> int:
    """Execute the record command.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Determine sources based on flags
    sources = []
    
    if args.mic:
        if not check_mic_available():
            print("Error: No microphone devices available", file=sys.stderr)
            return 1
        sources.append(SourceConfig(type='mic'))
        print(f"Recording from microphone for {args.seconds} seconds...")
        
    elif args.system:
        if not check_system_available():
            print("Error: No system audio loopback devices available", file=sys.stderr)
            return 1
        sources.append(SourceConfig(type='system'))
        print(f"Recording from system audio for {args.seconds} seconds...")
        
    elif args.both:
        mic_ok = check_mic_available()
        system_ok = check_system_available()
        
        if not mic_ok and not system_ok:
            print("Error: No audio devices available (mic or system)", file=sys.stderr)
            return 1
        
        sources_config = []
        if mic_ok:
            sources.append(SourceConfig(type='mic'))
            sources_config.append("microphone")
        if system_ok:
            sources.append(SourceConfig(type='system', gain=0.8))
            sources_config.append("system audio")
        
        print(f"Recording from { ' + '.join(sources_config) } for {args.seconds} seconds...")
        
    elif args.fake:
        fake_path = Path(args.fake)
        if not fake_path.exists():
            print(f"Error: Fake audio file not found: {fake_path}", file=sys.stderr)
            return 1
        sources.append(SourceConfig(type='fake', fake_path=str(fake_path), loop=False))
        print(f"Recording from fake source ({fake_path}) for {args.seconds} seconds...")
    
    # Create session config
    output_dir = Path(args.output_dir) if args.output_dir else None
    sample_rate = 16000
    max_frames = int(round(args.seconds * sample_rate))
    config = SessionConfig(
        sources=sources,
        output_dir=output_dir,
        sample_rate=sample_rate,
        channels=1,
        max_frames=max_frames,
    )
    
    # Start recording
    session = AudioSession()
    
    try:
        session.start(config)
    except AudioSourceError as e:
        print(f"Error starting audio source: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error starting recording: {e}", file=sys.stderr)
        return 1
    
    # Wait for recording duration
    try:
        time.sleep(args.seconds)
    except KeyboardInterrupt:
        print("\nInterrupted by user, stopping...")
    
    # Stop and finalize
    try:
        wav_path = session.stop()
    except Exception as e:
        print(f"Error stopping recording: {e}", file=sys.stderr)
        return 1
    
    # Report success
    stats = session.get_stats()
    print(f"\nRecording complete!")
    print(f"  WAV file: {wav_path}")
    print(f"  Duration: {stats.duration_seconds:.2f}s")
    print(f"  Frames recorded: {stats.frames_recorded}")
    
    return 0


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    if args.command == 'record':
        return cmd_record(args)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
