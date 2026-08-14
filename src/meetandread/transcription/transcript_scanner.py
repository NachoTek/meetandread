"""Transcript scanner for recording metadata.

Scans the recordings directory for saved .md transcript files, reads each
Transcript Footer through the canonical ``transcript_footer`` module, and
returns structured RecordingMeta objects for browsing and display in the
Library.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from meetandread.audio.storage.paths import get_recordings_dir
from meetandread.playback.bookmark import Bookmark, _parse_bookmark_entry
from meetandread.transcription import transcript_footer
from meetandread.transcription.retranscribe import RetranscribeRunner
from meetandread.transcription.transcript_footer import PostProcessOutcome

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
        bookmarks: List of Bookmark objects parsed from metadata (default empty)
        post_process_outcome: The durable Post-processing Outcome decoded from
            the Transcript Footer, or ``None`` when the Recording is Stalled
            (no Outcome recorded).
    """

    path: Path
    recording_time: str
    word_count: int
    speaker_count: int
    speakers: List[str]
    duration_seconds: float
    wav_exists: bool
    bookmarks: List[Bookmark] = field(default_factory=list)
    post_process_outcome: Optional[PostProcessOutcome] = None


def parse_metadata(md_path: Path) -> Optional[RecordingMeta]:
    """Parse a transcript .md file and extract recording metadata.

    Reads the file, decodes its Transcript Footer through the canonical
    ``transcript_footer.parse`` operation, and builds a RecordingMeta.

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

    data = transcript_footer.parse(text)
    if data is None:
        logger.warning("No metadata footer found in %s", md_path)
        return None

    # Extract fields
    recording_time: str = data.get("recording_start_time") or ""

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

    # Derive word_count by splitting each entry's text field on whitespace.
    # Word objects in metadata may contain multi-word text (e.g. when Whisper
    # returns segment-level data without word-level timestamps), so
    # len(words) can undercount. Splitting gives the true word count.
    words_data = data.get("words")
    if isinstance(words_data, list) and words_data:
        word_count: int = sum(
            len(w.get("text", "").split()) for w in words_data if w.get("text")
        )
    else:
        word_count: int = data.get("word_count", 0)

    # Check companion .wav file in the recordings directory (same stem).
    # Transcripts live in transcripts/ but their WAVs live in recordings/.
    recordings_dir = get_recordings_dir()
    wav_path = recordings_dir / f"{md_path.stem}.wav"
    wav_exists = wav_path.exists()

    # Parse bookmarks from metadata, skipping malformed entries
    raw_bookmarks = data.get("bookmarks", [])
    bookmarks: List[Bookmark] = []
    if isinstance(raw_bookmarks, list):
        for entry in raw_bookmarks:
            bm = _parse_bookmark_entry(entry)
            if bm is not None:
                bookmarks.append(bm)

    # Decode the durable Post-processing Outcome (None = Stalled).
    post_process_outcome = transcript_footer.outcome_from_block(
        data.get(transcript_footer.OUTCOME_KEY)
    )

    return RecordingMeta(
        path=md_path,
        recording_time=recording_time,
        word_count=word_count,
        speaker_count=len(speakers),
        speakers=speakers,
        duration_seconds=max_end_time,
        wav_exists=wav_exists,
        bookmarks=bookmarks,
        post_process_outcome=post_process_outcome,
    )


def scan_recordings(recordings_dir: Optional[Path] = None) -> List[RecordingMeta]:
    """Scan the transcripts directory for saved transcript files.

    Glob for ``*.md`` files, skip any ``*_enhanced.md`` (backwards compat),
    parse each one, and return a list sorted newest-first by recording_time.

    Args:
        recordings_dir: Directory to scan. Defaults to
            ``get_transcripts_dir()`` when None.

    Returns:
        List of RecordingMeta sorted by recording_time descending.
    """
    from meetandread.audio.storage.paths import get_transcripts_dir
    
    if recordings_dir is None:
        recordings_dir = get_transcripts_dir()

    if not recordings_dir.exists():
        logger.info("Recordings directory does not exist: %s", recordings_dir)
        return []

    results: List[RecordingMeta] = []
    md_files = sorted(recordings_dir.glob("*.md"))

    for md_path in md_files:
        # Skip legacy _enhanced.md files
        if md_path.name.endswith("_enhanced.md"):
            continue
        # Skip re-transcription sidecar files — these are temporary
        # comparison results waiting for Accept/Reject, not standalone
        # recordings. Match every sidecar tag (canonical ``_retranscribe_``
        # and legacy ``_scrub_``) so pre-rename sidecars stay hidden too.
        if RetranscribeRunner.is_sidecar_path(md_path):
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
