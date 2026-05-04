"""Tests for Unknown Speaker identity linking in history viewer.

Covers:
- "Unknown Speaker" renders as a clickable anchor (not just bold text)
- _link_speaker_identity_in_file updates words with speaker_id=None
- After linking, the transcript body shows the chosen identity name
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch

from meetandread.widgets.floating_panels import (
    _link_speaker_identity_in_file,
)


# ---------------------------------------------------------------------------
# Transcript file helpers
# ---------------------------------------------------------------------------

def _make_transcript_md(
    tmp_path: Path,
    words: list,
    speaker_matches: dict | None = None,
) -> Path:
    """Create a minimal transcript .md file with metadata footer.

    Uses the same format as TranscriptStore: **SpeakerLabel** headings
    followed by text, with JSON metadata footer.
    """
    # Group words by speaker_id to build segments/body
    segments = []
    current_speaker = "___UNSET___"
    seg_words = []
    for w in words:
        sid = w.get("speaker_id")
        if sid != current_speaker:
            if seg_words:
                segments.append({
                    "speaker_id": current_speaker,
                    "start_time": seg_words[0]["start_time"],
                    "end_time": seg_words[-1]["end_time"],
                })
            seg_words = []
            current_speaker = sid
        seg_words.append(w)
    if seg_words:
        segments.append({
            "speaker_id": current_speaker,
            "start_time": seg_words[0]["start_time"],
            "end_time": seg_words[-1]["end_time"],
        })

    # Build markdown body
    md_lines = ["# Transcript", ""]
    for seg in segments:
        label = seg["speaker_id"] if seg["speaker_id"] is not None else "Unknown Speaker"
        md_lines.append(f"**{label}**")
        md_lines.append("")
        seg_text = [w["text"] for w in words if w.get("speaker_id") == seg["speaker_id"]]
        if seg_text:
            md_lines.append(" ".join(seg_text))
            md_lines.append("")

    md_body = "\n".join(md_lines)

    metadata = {
        "words": words,
        "segments": segments,
        "speaker_matches": speaker_matches if speaker_matches is not None else {},
    }

    content = md_body + "\n---\n\n<!-- METADATA:\n" + json.dumps(metadata) + " -->\n"
    md_file = tmp_path / "transcript.md"
    md_file.write_text(content, encoding="utf-8")
    return md_file


def _read_metadata(md_path: Path) -> dict:
    """Parse metadata from a transcript .md file."""
    content = md_path.read_text(encoding="utf-8")
    # Try both formats: with and without newline after METADATA:
    for marker in ["\n---\n\n<!-- METADATA:\n", "\n---\n\n<!-- METADATA:"]:
        idx = content.find(marker)
        if idx != -1:
            raw = content[idx + len(marker):]
            # Strip trailing --> and whitespace
            end_marker = " -->"
            if raw.strip().endswith(end_marker):
                raw = raw.strip()[: -len(end_marker)]
            return json.loads(raw.strip())
    raise ValueError("No metadata footer found")


def _read_body(md_path: Path) -> str:
    """Read the markdown body (before metadata footer)."""
    content = md_path.read_text(encoding="utf-8")
    footer_marker = "\n---\n\n<!-- METADATA:"
    idx = content.find(footer_marker)
    assert idx != -1, "No metadata footer found"
    return content[:idx]


# ---------------------------------------------------------------------------
# Tests: _link_speaker_identity_in_file with __unknown__
# ---------------------------------------------------------------------------

class TestLinkUnknownSpeaker:
    """Test that __unknown__ sentinel correctly updates None speaker_ids."""

    def test_updates_none_speaker_id_words(self, tmp_path):
        """Words with speaker_id=None should get the chosen identity name."""
        words = [
            {"text": "Hello", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": None},
            {"text": "world", "start_time": 0.5, "end_time": 1.0, "confidence": 85, "speaker_id": None},
        ]
        md_path = _make_transcript_md(tmp_path, words)

        _link_speaker_identity_in_file(md_path, "__unknown__", "Alice")

        metadata = _read_metadata(md_path)
        assert all(w["speaker_id"] == "Alice" for w in metadata["words"])

    def test_updates_none_segments(self, tmp_path):
        """Segments with speaker_id=None should get the chosen identity name."""
        words = [
            {"text": "Hello", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": None},
        ]
        md_path = _make_transcript_md(tmp_path, words)

        _link_speaker_identity_in_file(md_path, "__unknown__", "Bob")

        metadata = _read_metadata(md_path)
        assert all(seg["speaker_id"] == "Bob" for seg in metadata["segments"])

    def test_updates_markdown_body_unknown_speaker(self, tmp_path):
        """The markdown body should replace **Unknown Speaker** with **Alice**."""
        words = [
            {"text": "Hello", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": None},
        ]
        md_path = _make_transcript_md(tmp_path, words)

        _link_speaker_identity_in_file(md_path, "__unknown__", "Alice")

        body = _read_body(md_path)
        assert "**Alice**" in body
        assert "**Unknown Speaker**" not in body

    def test_creates_speaker_matches_entry(self, tmp_path):
        """speaker_matches should get a __unknown__ entry with manual sentinel."""
        words = [
            {"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": None},
        ]
        md_path = _make_transcript_md(tmp_path, words)

        _link_speaker_identity_in_file(md_path, "__unknown__", "Carol")

        metadata = _read_metadata(md_path)
        assert "__unknown__" in metadata["speaker_matches"]
        assert metadata["speaker_matches"]["__unknown__"]["identity_name"] == "Carol"
        assert metadata["speaker_matches"]["__unknown__"]["confidence"] == "manual"
        assert metadata["speaker_matches"]["__unknown__"]["score"] == 1.0

    def test_mixed_speakers_only_updates_none(self, tmp_path):
        """When transcript has both SPK_0 and None, only None words are updated."""
        words = [
            {"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"},
            {"text": "there", "start_time": 0.5, "end_time": 1.0, "confidence": 85, "speaker_id": None},
        ]
        md_path = _make_transcript_md(tmp_path, words)

        _link_speaker_identity_in_file(md_path, "__unknown__", "Dave")

        metadata = _read_metadata(md_path)
        assert metadata["words"][0]["speaker_id"] == "SPK_0"  # unchanged
        assert metadata["words"][1]["speaker_id"] == "Dave"   # updated
        assert metadata["segments"][0]["speaker_id"] == "SPK_0"  # unchanged
        assert metadata["segments"][1]["speaker_id"] == "Dave"   # updated

    def test_no_signature_propagation_for_unknown(self, tmp_path):
        """Signature propagation should be skipped for __unknown__."""
        words = [
            {"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": None},
        ]
        md_path = _make_transcript_md(tmp_path, words)

        with patch("meetandread.widgets.floating_panels._propagate_identity_to_signatures") as mock_prop:
            _link_speaker_identity_in_file(md_path, "__unknown__", "Eve")
            mock_prop.assert_not_called()

    def test_empty_identity_name_is_noop(self, tmp_path):
        """Empty identity name should not modify the file."""
        words = [
            {"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": None},
        ]
        md_path = _make_transcript_md(tmp_path, words)
        original = md_path.read_text(encoding="utf-8")

        _link_speaker_identity_in_file(md_path, "__unknown__", "")

        assert md_path.read_text(encoding="utf-8") == original

    def test_identity_name_equals_raw_label_is_noop(self, tmp_path):
        """Identity name equal to raw_label should be a noop (only for real SPK labels)."""
        words = [
            {"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"},
        ]
        md_path = _make_transcript_md(tmp_path, words)
        original = md_path.read_text(encoding="utf-8")

        # SPK_0 renamed to SPK_0 should be noop
        _link_speaker_identity_in_file(md_path, "SPK_0", "SPK_0")

        assert md_path.read_text(encoding="utf-8") == original


# ---------------------------------------------------------------------------
# Tests: _render_history_transcript with Unknown Speaker
# ---------------------------------------------------------------------------

class TestRenderUnknownSpeakerAnchor:
    """Test that Unknown Speaker renders as clickable in history viewer."""

    @pytest.fixture
    def panel(self, qapp):
        from meetandread.widgets.floating_panels import FloatingTranscriptPanel
        panel = FloatingTranscriptPanel()
        yield panel
        panel.close()

    def test_unknown_speaker_renders_as_anchor(self, panel, tmp_path):
        """When words have speaker_id=None, 'Unknown Speaker' should be a clickable anchor."""
        words = [
            {"text": "Hello", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": None},
            {"text": "world", "start_time": 0.5, "end_time": 1.0, "confidence": 85, "speaker_id": None},
        ]
        md_path = _make_transcript_md(tmp_path, words)

        html = panel._render_history_transcript(md_path)
        assert html is not None, "Should render HTML even when all speakers are unknown"
        assert 'href="speaker:__unknown__"' in html
        assert "[Unknown Speaker]" in html

    def test_no_speakers_at_all_still_renders(self, panel, tmp_path):
        """Transcript with no speaker_id values at all should still render HTML."""
        words = [
            {"text": "Hello", "start_time": 0.0, "end_time": 0.5, "confidence": 90},
        ]
        md_path = _make_transcript_md(tmp_path, words)

        html = panel._render_history_transcript(md_path)
        assert html is not None, "Should render HTML even with no speakers"

    def test_mixed_known_and_unknown_both_clickable(self, panel, tmp_path):
        """Both known (SPK_0) and unknown speakers should be clickable anchors."""
        words = [
            {"text": "Hi", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": "SPK_0"},
            {"text": "there", "start_time": 0.5, "end_time": 1.0, "confidence": 85, "speaker_id": None},
        ]
        md_path = _make_transcript_md(tmp_path, words)

        html = panel._render_history_transcript(md_path)
        assert html is not None
        assert 'href="speaker:SPK_0"' in html
        assert 'href="speaker:__unknown__"' in html

    def test_unknown_speaker_after_linking_shows_name(self, panel, tmp_path):
        """After linking Unknown Speaker to an identity, the name should appear."""
        words = [
            {"text": "Hello", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": None},
        ]
        md_path = _make_transcript_md(tmp_path, words)

        # First link it
        _link_speaker_identity_in_file(md_path, "__unknown__", "Frank")

        # Now render — should show [Frank] as clickable
        html = panel._render_history_transcript(md_path)
        assert html is not None
        assert "[Frank]" in html
        assert "Unknown Speaker" not in html


class TestSettingsPanelUnknownSpeakerAnchor:
    """Test that the Settings panel also renders Unknown Speaker as clickable."""

    @pytest.fixture
    def settings_panel(self, qapp):
        from meetandread.widgets.floating_panels import FloatingSettingsPanel
        panel = FloatingSettingsPanel()
        yield panel
        panel.close()

    def test_unknown_speaker_clickable_in_settings(self, settings_panel, tmp_path):
        """Settings panel history should also make Unknown Speaker clickable."""
        words = [
            {"text": "Hello", "start_time": 0.0, "end_time": 0.5, "confidence": 90, "speaker_id": None},
        ]
        md_path = _make_transcript_md(tmp_path, words)

        html = settings_panel._render_history_transcript(md_path)
        assert html is not None, "Settings panel should render HTML for unknown speakers"
        assert 'href="speaker:__unknown__"' in html
        assert "[Unknown Speaker]" in html
