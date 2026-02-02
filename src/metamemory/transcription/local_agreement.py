"""Local agreement buffer for transcription deduplication.

Prevents text flickering and duplication in streaming transcription by only
committing text after N consecutive iterations agree on the same content.

Based on whisper_streaming implementation pattern.
"""

from typing import Optional


class LocalAgreementBuffer:
    """Buffer that prevents text flickering through local agreement policy.
    
    Instead of emitting text immediately (which causes flickering), this buffer
    tracks recent transcriptions and only commits text when N consecutive
    iterations produce the same prefix. This stabilizes streaming output.
    
    The algorithm works by:
    1. Maintaining a buffer of the current agreed-upon text
    2. Finding common prefix between buffer and each new transcription
    3. Tracking how many consecutive iterations have agreed on each position
    4. Only committing text that has been stable for N iterations
    
    Key insight: When new content is added (buffer grows), that new content
    must survive N iterations of exact matching before being committed.
    
    Reference: RESEARCH.md Pattern 2 (line 103-144) and whisper_streaming
    
    Example:
        buffer = LocalAgreementBuffer(agreement_threshold=2)
        
        # Build agreement on base text
        buffer.process_iteration('Hello world')  # Buffer: 'Hello world'
        buffer.process_iteration('Hello world')  # Agreement: 1
        buffer.process_iteration('Hello world')  # Agreement: 2 -> commit 'Hello world'
        
        # Extension needs fresh agreement
        buffer.process_iteration('Hello world today')  # Buffer extends, reset
        buffer.process_iteration('Hello world today')  # Agreement: 1
        buffer.process_iteration('Hello world today')  # Agreement: 2 -> commit ' today'
    """
    
    def __init__(self, agreement_threshold: int = 2):
        """Initialize the local agreement buffer.
        
        Args:
            agreement_threshold: Number of consecutive agreements required
                               before committing text (default 2)
        """
        self.agreement_threshold = agreement_threshold
        
        # Committed text (stable, shown to user)
        self._committed_text = ""
        
        # Current buffer content
        self._buffer = ""
        
        # Counter for consecutive agreements on current buffer
        self._agreement_count = 0
        
        # Track buffer length at time of last commit
        self._last_commit_len = 0
        
        # Track buffer length when content was last added/changed
        # Content beyond this point needs fresh agreement
        self._stable_len = 0
    
    def process_iteration(self, new_transcription: str) -> str:
        """Process a new transcription and return newly committed text.
        
        Args:
            new_transcription: New transcription text from model
            
        Returns:
            Newly committed text (empty string if nothing new to commit)
        """
        # First transcription - just store it
        if not self._buffer:
            self._buffer = new_transcription
            self._agreement_count = 0
            self._stable_len = len(new_transcription)
            return ""
        
        # Find common prefix
        common = self._common_prefix(self._buffer, new_transcription)
        
        # Check if this is an exact match of current buffer
        is_exact_match = (new_transcription == self._buffer)
        
        # Check if new transcription extends the buffer
        is_extension = len(new_transcription) > len(self._buffer) and common == self._buffer
        
        if is_extension:
            # Buffer is being extended with new content
            # The previous content up to _stable_len is agreed upon
            # The new extension needs fresh agreement
            self._buffer = new_transcription
            self._agreement_count = 1  # This iteration counts as first agreement
            # stable_len stays at current position (extension is not yet stable)
        elif is_exact_match:
            # Exact match - increment agreement
            self._agreement_count += 1
            # Don't update stable_len yet - it happens after commit check
            # This ensures we only commit what was stable BEFORE this match
        else:
            # Content diverged (common prefix shorter than buffer)
            # Roll back to common prefix and reset agreement
            self._buffer = common
            self._agreement_count = 1
            self._stable_len = len(common)
        
        # Check if we can commit new content
        # We can only commit content that is both:
        # 1. Beyond last_commit_len (not yet committed)
        # 2. Within stable_len (has reached agreement threshold)
        committed_now = ""
        if self._agreement_count >= self.agreement_threshold:
            # Can commit up to stable_len, but only what hasn't been committed yet
            commit_up_to = min(self._stable_len, len(self._buffer))
            if commit_up_to > self._last_commit_len:
                committed_now = self._buffer[self._last_commit_len:commit_up_to]
                self._committed_text = self._buffer[:commit_up_to]
                self._last_commit_len = commit_up_to
                
                # Reset agreement for next batch
                self._agreement_count = 0
        
        # Now update stable_len for exact matches
        # (This happens AFTER commit check so new content doesn't immediately become commitable)
        if is_exact_match:
            self._stable_len = len(self._buffer)
        
        return committed_now
    
    def get_committed(self) -> str:
        """Get all committed text so far."""
        return self._committed_text
    
    def get_pending(self) -> str:
        """Get uncommitted (volatile) text."""
        if len(self._buffer) > len(self._committed_text):
            return self._buffer[len(self._committed_text):]
        return ""
    
    def get_buffer(self) -> str:
        """Get current buffer content."""
        return self._buffer
    
    def reset(self) -> None:
        """Clear all buffers and reset state."""
        self._committed_text = ""
        self._buffer = ""
        self._agreement_count = 0
        self._last_commit_len = 0
        self._stable_len = 0
    
    def _common_prefix(self, a: str, b: str) -> str:
        """Find common prefix between two strings."""
        min_len = min(len(a), len(b))
        for i in range(min_len):
            if a[i] != b[i]:
                return a[:i]
        return a[:min_len]
