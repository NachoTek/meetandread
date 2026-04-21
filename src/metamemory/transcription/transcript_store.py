"""Transcript storage with word-level tracking.

Provides in-memory transcript storage with rich metadata for each word,
enabling word-by-word display, confidence color coding, and future
enhancement features.
"""

import threading
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path


@dataclass
class Word:
    """A single word in the transcript with metadata.
    
    Attributes:
        text: The word text
        start_time: Start timestamp in seconds from recording start
        end_time: End timestamp in seconds
        confidence: Confidence score (0-100)
        speaker_id: Optional speaker identifier (Phase 4)
    """
    text: str
    start_time: float
    end_time: float
    confidence: int
    speaker_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "confidence": self.confidence,
            "speaker_id": self.speaker_id,
        }


@dataclass
class Segment:
    """A segment of contiguous words from a single speaker.
    
    Attributes:
        words: List of Word objects in this segment
        start_time: Segment start timestamp
        end_time: Segment end timestamp
        avg_confidence: Average confidence score (0-100)
        speaker_id: Speaker identifier for this segment
    """
    words: List[Word] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    avg_confidence: int = 0
    speaker_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "words": [w.to_dict() for w in self.words],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "avg_confidence": self.avg_confidence,
            "speaker_id": self.speaker_id,
        }


class TranscriptStore:
    """In-memory storage for transcript with word-level data.
    
    Thread-safe storage for transcribed words with timestamps and confidence.
    Supports word-by-word streaming display, confidence color coding,
    and future speaker identification features.
    
    Memory Management:
    - For 30+ minute recordings: ~5000 words, ~500KB memory
    - Each word is small (~100 bytes)
    - Acceptable for in-memory storage
    - Future: may trim old words if memory becomes concern
    
    Thread Safety:
    - All operations are thread-safe via internal locking
    - add_words can be called from transcription thread
    - get_segments can be called from UI thread
    
    Example:
        store = TranscriptStore()
        
        # Add words from transcription
        store.add_words([
            Word("Hello", 0.0, 0.5, 85),
            Word("world", 0.5, 1.0, 92)
        ])
        
        # Get segments for display
        segments = store.get_segments(since_time=0)
        
        # Export to markdown
        markdown = store.to_markdown()
        
        # Reset for new recording
        store.clear()
    """
    
    def __init__(self):
        """Initialize empty transcript store."""
        self._words: List[Word] = []
        self._lock = threading.Lock()
        self._recording_start_time: Optional[datetime] = None
        self._last_segment_time: float = 0.0
    
    def start_recording(self) -> None:
        """Mark the start of a new recording session."""
        with self._lock:
            self._recording_start_time = datetime.now()
            self._last_segment_time = 0.0
    
    def add_words(self, words: List[Word]) -> None:
        """Add new words to the transcript.
        
        Thread-safe - can be called from transcription thread.
        
        Args:
            words: List of Word objects to add
        """
        with self._lock:
            self._words.extend(words)
            if words:
                self._last_segment_time = max(w.end_time for w in words)
    
    def get_segments(self, since_time: float = 0) -> List[Segment]:
        """Get segments after a specific time.
        
        Thread-safe - can be called from UI thread.
        
        Args:
            since_time: Return segments with start_time >= this value
        
        Returns:
            List of Segment objects
        """
        with self._lock:
            # Group words by speaker_id into segments
            segments = []
            current_segment_words = []
            current_speaker = None
            
            for word in self._words:
                if word.start_time < since_time:
                    continue
                
                # Start new segment if speaker changes
                if word.speaker_id != current_speaker:
                    if current_segment_words:
                        segments.append(self._create_segment(current_segment_words))
                    current_segment_words = []
                    current_speaker = word.speaker_id
                
                current_segment_words.append(word)
            
            # Add final segment
            if current_segment_words:
                segments.append(self._create_segment(current_segment_words))
            
            return segments
    
    def get_all_words(self) -> List[Word]:
        """Get all words in chronological order.
        
        Returns:
            List of all Word objects
        """
        with self._lock:
            return list(self._words)
    
    def get_recent_words(self, count: int) -> List[Word]:
        """Get the last N words.
        
        Args:
            count: Number of words to return
        
        Returns:
            List of the most recent Word objects
        """
        with self._lock:
            return list(self._words[-count:]) if count > 0 else []
    
    def get_word_count(self) -> int:
        """Get total number of words in transcript.
        
        Returns:
            Word count
        """
        with self._lock:
            return len(self._words)
    
    def clear(self) -> None:
        """Reset transcript for new recording."""
        with self._lock:
            self._words = []
            self._recording_start_time = None
            self._last_segment_time = 0.0
    
    def to_markdown(self, include_confidence: bool = True,
                   include_timestamps: bool = True) -> str:
        """Export transcript to markdown format.
        
        Args:
            include_confidence: Include confidence scores
            include_timestamps: Include timestamps
        
        Returns:
            Markdown-formatted transcript string
        """
        with self._lock:
            lines = []
            lines.append("# Transcript")
            lines.append("")
            
            if self._recording_start_time:
                lines.append(f"**Recorded:** {self._recording_start_time.isoformat()}")
                lines.append("")
            
            # Group words into segments by speaker and time gaps
            segments = self._get_segments_internal()
            
            for segment in segments:
                # Speaker label
                speaker = segment.speaker_id or "Unknown Speaker"
                lines.append(f"**{speaker}**")
                lines.append("")
                
                # Format text with optional metadata
                text_parts = []
                for word in segment.words:
                    if include_confidence and word.confidence < 80:
                        # Show low confidence words with marker
                        text_parts.append(f"{word.text} ({word.confidence}%)")
                    else:
                        text_parts.append(word.text)
                
                text = " ".join(text_parts)
                lines.append(text)
                lines.append("")
                
                if include_timestamps:
                    start = self._format_timestamp(segment.start_time)
                    end = self._format_timestamp(segment.end_time)
                    lines.append(f"*[{start} - {end}]*")
                    lines.append("")
            
            return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize transcript to dictionary.
        
        Returns:
            Dictionary with all transcript data
        """
        with self._lock:
            return {
                "recording_start_time": self._recording_start_time.isoformat() 
                    if self._recording_start_time else None,
                "word_count": len(self._words),
                "words": [w.to_dict() for w in self._words],
                "segments": [s.to_dict() for s in self._get_segments_internal()],
            }
    
    def save_to_file(self, path: Path) -> None:
        """Save transcript to a file.
        
        Saves as markdown with embedded JSON metadata.
        
        Args:
            path: Path to save the transcript
        """
        markdown = self.to_markdown()
        data = self.to_dict()
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(markdown)
            f.write("\n\n---\n\n")
            f.write("<!-- METADATA: ")
            import json
            f.write(json.dumps(data, indent=2))
            f.write(" -->\n")
    
    def _create_segment(self, words: List[Word]) -> Segment:
        """Create a Segment from a list of words.
        
        Args:
            words: List of words in the segment
        
        Returns:
            Segment object
        """
        if not words:
            return Segment()
        
        avg_confidence = sum(w.confidence for w in words) // len(words)
        
        return Segment(
            words=list(words),
            start_time=words[0].start_time,
            end_time=words[-1].end_time,
            avg_confidence=avg_confidence,
            speaker_id=words[0].speaker_id
        )
    
    def _get_segments_internal(self) -> List[Segment]:
        """Internal method to get all segments (assumes lock is held)."""
        segments = []
        current_segment_words = []
        current_speaker = None
        
        for word in self._words:
            # Start new segment if speaker changes
            if word.speaker_id != current_speaker:
                if current_segment_words:
                    segments.append(self._create_segment(current_segment_words))
                current_segment_words = []
                current_speaker = word.speaker_id
            
            current_segment_words.append(word)
        
        # Add final segment
        if current_segment_words:
            segments.append(self._create_segment(current_segment_words))
        
        return segments
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as MM:SS.
        
        Args:
            seconds: Time in seconds
        
        Returns:
            Formatted timestamp string
        """
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"
