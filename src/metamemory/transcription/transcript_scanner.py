"""Transcript scanner for recording metadata.

Scans the recordings directory for saved .md transcript files, parses the
embedded JSON metadata footer, and returns structured RecordingMeta objects
for browsing and display in the History tab.

METADATA FORMAT (written by TranscriptStore.save_to_file):
    Markdown content
    ...
    ---
    <!-- METADATA: { "recording_start_time": "...", "word_count": N, "words": [...] } -->
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from metamemory.audio.storage.paths import get_recordings_dir

logger = logging.getLogger(__name__)


@dataclass
class RecordingMeta:
    """Structured metadata for a single saved recording.

    Attributes:
        path: Path to the .md transcript file
        recording_time: ISO timestamp from metadata
        word_count: Total word count
        speaker_count: Number of unique speakers
        speakers: List of unique speaker IDs
        duration_seconds: Derived from max end_time across all words (0.0 if none)
        wav_exists: Whether a corresponding .wav file exists
    """

    path: Path
    recording_time: str
    word_count: int
    speaker_count: int
    speakers: List[str]
    duration_seconds: float
    wav_exists: bool


def parse_metadata(md_path: Path) -> Optional[RecordingMeta]:
    """Parse a transcript .md file and extract recording metadata.

    Reads the file, locates the ``<!-- METADATA: ... -->`` footer, parses
    the embedded JSON, and builds a RecordingMeta.

    Args:
        md_path: Path to a saved transcript .md file.

    Returns:
        RecordingMeta on success, or None if the file has no metadata
        footer or the JSON is malformed (logs a warning).
    """
    try:
        text = md_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Cannot read transcript file %s: %s", md_path, exc)
        return None

    # Find the metadata footer — may span multiple lines when json.dumps(indent=2)
    # Format: <!-- METADATA: { ... JSON ... } -->
    prefix = "<!-- METADATA: "
    suffix = " -->"
    metadata_start: Optional[int] = None
    metadata_end: Optional[int] = None

    lines = text.splitlines()
    for i, line in enumerate(lines):
        if metadata_start is None and line.strip().startswith(prefix):
            metadata_start = i
        if metadata_start is not None and line.strip().endswith(suffix):
            metadata_end = i
            break

    if metadata_start is None or metadata_end is None:
        logger.warning("No metadata footer found in %s", md_path)
        return None

    # Reconstruct the metadata block and strip the prefix/suffix markers
    block = "\n".join(lines[metadata_start : metadata_end + 1])

    # Strip prefix from first line content
    prefix_idx = block.index(prefix)
    json_str = block[prefix_idx + len(prefix) :]

    # Strip suffix from last part
    if json_str.rstrip().endswith(suffix):
        # Find the last occurrence of suffix
        json_str = json_str.rstrip()
        json_str = json_str[: -len(suffix)]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.warning("Malformed metadata JSON in %s: %s", md_path, exc)
        return None

    # Extract fields
    recording_time: str = data.get("recording_start_time") or ""
    word_count: int = data.get("word_count", 0)

    # Collect unique speakers from words array
    speakers: List[str] = []
    seen_speakers = set()
    max_end_time = 0.0

    for word in data.get("words", []):
        sid = word.get("speaker_id")
        if sid is not None and sid not in seen_speakers:
            seen_speakers.add(sid)
            speakers.append(sid)
        end = word.get("end_time", 0.0)
        if isinstance(end, (int, float)) and end > max_end_time:
            max_end_time = end

    # Check companion .wav file
    wav_path = md_path.with_suffix(".wav")
    wav_exists = wav_path.exists()

    return RecordingMeta(
        path=md_path,
        recording_time=recording_time,
        word_count=word_count,
        speaker_count=len(speakers),
        speakers=speakers,
        duration_seconds=max_end_time,
        wav_exists=wav_exists,
    )


def scan_recordings(recordings_dir: Optional[Path] = None) -> List[RecordingMeta]:
    """Scan the recordings directory for saved transcript files.

    Glob for ``*.md`` files, skip any ``*_enhanced.md`` (backwards compat),
    parse each one, and return a list sorted newest-first by recording_time.

    Args:
        recordings_dir: Directory to scan. Defaults to
            ``get_recordings_dir()`` when None.

    Returns:
        List of RecordingMeta sorted by recording_time descending.
    """
    if recordings_dir is None:
        recordings_dir = get_recordings_dir()

    if not recordings_dir.exists():
        logger.info("Recordings directory does not exist: %s", recordings_dir)
        return []

    results: List[RecordingMeta] = []
    md_files = sorted(recordings_dir.glob("*.md"))

    for md_path in md_files:
        # Skip legacy _enhanced.md files
        if md_path.name.endswith("_enhanced.md"):
            continue

        meta = parse_metadata(md_path)
        if meta is not None:
            results.append(meta)

    # Sort newest-first by recording_time descending
    results.sort(key=lambda m: m.recording_time, reverse=True)

    logger.info(
        "Scanned %d .md files, found %d valid transcripts", len(md_files), len(results)
    )

    return results
